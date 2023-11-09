# ---
# output-directory: "/tmp/stable-diffusion-xl"
# args: ["--prompt", "An astronaut riding a green horse"]
# runtimes: ["runc", "gvisor"]
# ---
# # Stable Diffusion XL 1.0
#
# This example is similar to the [Stable Diffusion CLI](/docs/examples/stable_diffusion_cli)
# example, but it generates images from the larger SDXL 1.0 model. Specifically, it runs the
# first set of steps with the base model, followed by the refiner model.
#
# [Try out the live demo here!](https://modal-labs--stable-diffusion-xl-app.modal.run/) The first
# generation may include a cold-start, which takes around 20 seconds. The inference speed depends on the GPU
# and step count (for reference, an A100 runs 40 steps in 8 seconds).

# ## Basic setup

from pathlib import Path
import pathlib
import tempfile
from modal import Image, Mount, Stub, asgi_app, gpu, method, Secret, Volume
import datetime

s3_secret = Secret.from_name("anj-aws-secret")

# ## Define a container image
#
# To take advantage of Modal's blazing fast cold-start times, we'll need to download our model weights
# inside our container image with a download function. We ignore binaries, ONNX weights and 32-bit weights.
#
# Tip: avoid using global variables in this function to ensure the download step detects model changes and
# triggers a rebuild.


def download_models():
    from huggingface_hub import snapshot_download

    ignore = ["*.bin", "*.onnx_data", "*/diffusion_pytorch_model.safetensors"]
    snapshot_download(
        "stabilityai/stable-diffusion-xl-base-1.0", ignore_patterns=ignore
    )
    snapshot_download(
        "stabilityai/stable-diffusion-xl-refiner-1.0", ignore_patterns=ignore
    )


image = (
    Image.debian_slim()
    .apt_install(
        "libglib2.0-0", "libsm6", "libxrender1", "libxext6", "ffmpeg", "libgl1"
    )
    # .apt_install("sqlite3")
    .pip_install(
        "diffusers~=0.19",
        "invisible_watermark~=0.1",
        "transformers~=4.31",
        "accelerate~=0.21",
        "safetensors~=0.3",
    )
    .pip_install("boto3")
    .pip_install("datasette~=0.63.2", "sqlite-utils")
    .pip_install("Jinja2~=3.0.1")
    .pip_install("jinja2")
    .run_function(download_models)
)

stub = Stub("stable-diffusion-xl", image=image)
BUCKET_NAME = "anj-image-gen-images"

stub.volume = Volume.persisted("image-gen-cache-vol")

VOLUME_DIR = "/cache-vol"
DB_PATH = pathlib.Path(VOLUME_DIR, "image-gen.db")

# ## Load model and run inference
#
# The container lifecycle [`__enter__` function](https://modal.com/docs/guide/lifecycle-functions#container-lifecycle-beta)
# loads the model at startup. Then, we evaluate it in the `run_inference` function.
#
# To avoid excessive cold-starts, we set the idle timeout to 240 seconds, meaning once a GPU has loaded the model it will stay
# online for 4 minutes before spinning down. This can be adjusted for cost/experience trade-offs.


# TODO: Can speed up inference by switching too A100
@stub.cls(
    gpu=gpu.A10G(),
    secrets=[s3_secret],
    container_idle_timeout=240,
    volumes={VOLUME_DIR: stub.volume},
)
class Model:
    # Define your S3 Bucket name, just the name, not the ARN or URL
    def __enter__(self):
        import torch
        from diffusers import DiffusionPipeline

        load_options = dict(
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            device_map="auto",
        )

        # Load base model
        self.base = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", **load_options
        )

        # Load refiner model
        self.refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=self.base.text_encoder_2,
            vae=self.base.vae,
            **load_options,
        )

        # Setup db #####################
        import sqlite_utils

        # Create the database if it doesn't exist
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite_utils.Database(DB_PATH)
        self.table = self.db["image-gen"]
        self.table.create(
            {
                "id": int,
                "image_s3": str,
                "user": str,
                "prompt": str,
                "metadata": str,
                "created_at": str,
                "updated_at": str,
            },
            pk="id",
            if_not_exists=True,
        )  # , foreign_keys=("user"))
        self.table.create_index(["image_s3"], if_not_exists=True)
        self.table.create_index(["user"], if_not_exists=True)
        self.table.create_index(["prompt"], if_not_exists=True)
        self.table.create_index(["metadata"], if_not_exists=True)
        self.db.close()
        # Sync db with volume (cache)
        # stub.volume.commit()
        #################################

        # Compiling the model graph is JIT so this will increase inference time for the first run
        # but speed up subsequent runs. Uncomment to enable.
        # self.base.unet = torch.compile(self.base.unet, mode="reduce-overhead", fullgraph=True)
        # self.refiner.unet = torch.compile(self.refiner.unet, mode="reduce-overhead", fullgraph=True)

    # @method()
    def push_img_to_db(self, image_s3, prompt, id=None, user=None, metadata=None):
        import sqlite_utils

        self.db = sqlite_utils.Database(DB_PATH)
        self.table = self.db["image-gen"]
        data = {
            "image_s3": image_s3,
            "user": user,
            "prompt": prompt,
            "metadata": metadata,
            "updated_at": str(datetime.datetime.now()),
        }

        image_id = None
        if id is not None:  # Update existing record
            self.db["image-gen"].upsert(
                data, pk="id"
            )  # Assuming 'id' is the primary key
            image_id = id
        else:  # Insert new record
            data["created_at"] = str(datetime.datetime.now())
            image_id = self.db["image-gen"].insert(data).last_pk

        self.db.close()
        # Sync db with volume (cache)
        stub.volume.commit()

        return image_id

    @method()
    def get_img_from_db(self, image_id):
        from PIL import Image
        from sqlite3 import connect
        import boto3
        import sqlite_utils
        import io
        import os
        import base64
        from fastapi.responses import Response, JSONResponse

        self.db = sqlite_utils.Database(DB_PATH)

        record = self.db["image-gen"].get(image_id)

        # If the row is found, proceed to download from S3
        if record:
            image_s3 = record["image_s3"]
            metadata = record["metadata"]
            created_at = record["created_at"]
            prompt = record["prompt"]
            print(image_s3, metadata, created_at, prompt)
            s3_client = boto3.client("s3")

            # Download the image file from S3 into the buffer
            with tempfile.TemporaryDirectory() as td:
                full_path = os.path.join(td, image_s3)
                with open(full_path, "wb") as tmp_file:
                    s3_client.download_fileobj(
                        BUCKET_NAME,
                        image_s3,
                        tmp_file,
                    )
                # Convert to an image
                with open(full_path, "rb") as image_file:
                    with Image.open(image_file) as img:
                        byte_stream = io.BytesIO()
                        img.save(byte_stream, format="PNG")
                        
            image_bytes = byte_stream.getvalue()
            encoded_image = base64.b64encode(image_bytes).decode("utf-8")
            # print("Encoded image: ", encoded_image)
            content={
                "id": image_id,
                "s3": image_s3,
                "metadata": metadata,
                "created_at": created_at,
                "prompt": prompt,
                "image": encoded_image,
            }
            # return JSONResponse(
            #     content
            # )
            return content
            # return image_bytes
        else:
            # Handle the case where there is no record with the given ID
            print("No matching record found for image id: ", image_id)
            return None

    @method()
    def download_from_s3(self, bucket_name, s3_key, image_bytes_buffer):
        import boto3

        """
        Download a file from S3 to the given local path.

        :param bucket_name: The name of the S3 bucket.
        :param s3_key: The key of the file in the S3 bucket.
        :param local_path: The local path where the file should be saved.
        """
        s3_client = boto3.client("s3")

        # local_path = str(Path("/tmp/stable-diffusion-xl") / s3_key)
        # Path(local_path).parent.mkdir(parents=True, exist_ok=True)

        # s3_client.download_file(bucket_name, s3_key, local_path)
        ret = s3_client.download_fileobj(bucket_name, s3_key, image_bytes_buffer)
        print("S3 ret: ", ret)
        # print(f"File downloaded successfully from S3: {s3_key}")

        # return local_path

    @method()
    def upload_to_s3(self, image_bytes):
        import os
        import boto3
        from botocore.exceptions import NoCredentialsError
        import uuid

        # Ensure the temp directory exists
        import os

        # Ensure the temp directory exists
        temp_dir = "temp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # Generate a unique file name
        filename = f"{uuid.uuid4()}.png"
        temp_file_path = os.path.join(temp_dir, filename)

        # Save the image temporarily
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(image_bytes)

        s3_client = boto3.client("s3")
        # s3_client.create_bucket(Bucket=BUCKET_NAME)
        ret = s3_client.upload_file(temp_file_path, BUCKET_NAME, filename)
        print("S3 ret: ", ret)
        file_url = f"https://{BUCKET_NAME}.s3.amazonaws.com/{filename}"
        print("Uploaded to S3: ", filename)

        return filename

    @method()
    def inference(
        self, prompt, n_steps=24, high_noise_frac=0.8
    ):  # Change n_steps to 24
        negative_prompt = "disfigured, ugly, deformed, poor details, bad anatomy, blurry"
        image = self.base(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=n_steps,
            denoising_end=high_noise_frac,
            output_type="latent",
        ).images
        image = self.refiner(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=n_steps,
            denoising_start=high_noise_frac,
            image=image,
        ).images[0]

        import io

        byte_stream = io.BytesIO()
        image.save(byte_stream, format="PNG")
        image_bytes = byte_stream.getvalue()

        return image_bytes


# And this is our entrypoint; where the CLI is invoked. Explore CLI options
# with: `modal run stable_diffusion_xl.py --prompt 'An astronaut riding a green horse'`


@stub.local_entrypoint()
def main(prompt: str):
    print("start main")
    image_bytes = Model().inference.remote(prompt)
    s3_key = Model().upload_to_s3.remote(image_bytes)
    # import sqlite_utils
    # # Create the database if it doesn't exist
    # db = sqlite_utils.Database(DB_PATH)
    # table = db["image-gen"]
    # table.insert({
    #     "image_s3": s3_key,
    #     "user": "1",
    #     "prompt": prompt,
    #     "metadata": "{}",
    #     "created_at": str(datetime.datetime.now()),
    #     "updated_at": str(datetime.datetime.now()),
    # })
    # db.close()
    # #Sync db with volume (cache)
    # stub.volume.commit()

    dir = Path("/tmp/stable-diffusion-xl")
    if not dir.exists():
        dir.mkdir(exist_ok=True, parents=True)
    output_path = dir  # / "output.png"
    Model().download_from_s3.remote(
        BUCKET_NAME, "c2fb43cd-ab56-4b26-a3c6-a027ab1b54b7.png", output_path
    )


# ## A user interface
#
# Here we ship a simple web application that exposes a front-end (written in Alpine.js) for
# our backend deployment.
#
# The Model class will serve multiple users from a its own shared pool of warm GPU containers automatically.
#
# We can deploy this with `modal deploy stable_diffusion_xl.py`.

frontend_path = Path(__file__).parent / "frontend"


@stub.function(
    mounts=[Mount.from_local_dir(frontend_path, remote_path="/assets")],
    allow_concurrent_inputs=20,
    cpu=1,  # default is 0.1, make 1 on live deploy
    volumes={VOLUME_DIR: stub.volume},
)
@asgi_app()
def app():
    import fastapi.staticfiles
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse
    from sqlite3 import connect
    from datasette.app import Datasette

    web_app = FastAPI()
    ds = Datasette(files=[DB_PATH], settings={"sql_time_limit_ms": 10000})

    @web_app.get("/save-new-img/{prompt}")
    async def save_new_image(prompt: str):
        from fastapi.responses import Response, JSONResponse

        image_id = Model().push_img_to_db(None, prompt)
        print(image_id)

        return {"id": image_id}

    @web_app.get("/infer/{prompt}")
    async def infer(prompt: str):
        from fastapi.responses import Response, JSONResponse
        import base64

        image_bytes = Model().inference.remote(prompt)

        s3_key = Model().upload_to_s3.remote(image_bytes)
        image_id = Model().push_img_to_db(s3_key, prompt)

        # return Response(image_bytes, media_type="image/png")
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        return JSONResponse(content={"id": image_id, "image": encoded_image})

    @web_app.get("/image/{id}", response_class=HTMLResponse)
    async def image(request: Request, id: str):
        from fastapi.templating import Jinja2Templates
        templates = Jinja2Templates(directory="/assets")
        response_obj = Model().get_img_from_db.remote(id)
        response_obj['request'] = request
        return templates.TemplateResponse("index.html", response_obj)

    web_app.mount("/", fastapi.staticfiles.StaticFiles(directory="/assets", html=True))

    return web_app
