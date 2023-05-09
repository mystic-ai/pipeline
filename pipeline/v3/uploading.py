# import hashlib
# import io
# import os

# from tqdm import tqdm

# from pipeline.objects.variable import PipelineFile
# from pipeline.util import CallbackBytesIO
# from pipeline.util.logging import PIPELINE_FILE_STR

# FILE_CHUNK_SIZE = 200 * 1024 * 1024  # 200 MiB
# BINARY_MIME_TYPE = "application/octet-stream"


# def _initialise_direct_pipeline_file_upload(self, file_size: int) -> str:
#     """Initialise a direct multi-part pipeline file upload"""
#     file_size = file_size

#     response = self._post(
#         "/v2/pipeline-files/initiate-multipart-upload",
#         json_data={
#             "file_size": file_size,
#         },
#     )

#     direct_upload_get = PipelineFileDirectUploadInitGet.parse_obj(response)
#     return direct_upload_get.pipeline_file_id


# def _direct_upload_pipeline_file_chunk(
#     self,
#     data: Union[io.BytesIO, CallbackIOWrapper],
#     pipeline_file_id: str,
#     part_num: int,
# ) -> MultipartUploadMetadata:
#     """Upload a single chunk of a multi-part pipeline file upload.

#     Returns the metadata associated with this upload (this is needed to pass into
#     the finalisation step).
#     """
#     # get presigned URL
#     part_upload_schema = PipelineFileDirectUploadPartCreate(
#         pipeline_file_id=pipeline_file_id, part_num=part_num
#     )
#     response = self._post(
#         "/v2/pipeline-files/presigned-url",
#         json_data=part_upload_schema.dict(),
#     )
#     part_upload_get = PipelineFileDirectUploadPartGet.parse_obj(response)
#     # upload file chunk
#     response = httpx.put(
#         part_upload_get.upload_url,
#         content=data,
#         timeout=self._timeout,
#     )
#     response.raise_for_status()
#     etag = response.headers["ETag"]
#     return MultipartUploadMetadata(ETag=etag, PartNumber=part_num)


# def _finalise_direct_pipeline_file_upload(
#     self, pipeline_file_id: str, multipart_metadata: List[MultipartUploadMetadata]
# ) -> PipelineFileGet:
#     """Finalise the direct multi-part pipeline file upload"""
#     finalise_upload_schema = PipelineFileDirectUploadFinaliseCreate(
#         pipeline_file_id=pipeline_file_id,
#         multipart_metadata=multipart_metadata,
#     )
#     response = self._post(
#         "/v2/pipeline-files/finalise-multipart-upload",
#         json_data=finalise_upload_schema.dict(),
#     )
#     return PipelineFileGet.parse_obj(response)


# def upload_pipeline_file(self, pipeline_file: PipelineFile):
#     """Upload PipelineFile given by pipeline_file.

#     Since PipelineFiles can be very large, we implement this slightly
#     differently to regular file uploads:
#     - We need to split the file into chunks based on FILE_CHUNK_SIZE
#     - We first initialise the multi-part upload with the server
#     - We then upload the file in chunks (requesting a presigned upload URL for each
#         chunk beforehand)
#     - Lastly, we finalise the multi-part upload with the server
#     """

#     file_size = os.path.getsize(pipeline_file.path)

#     pipeline_file_id = _initialise_direct_pipeline_file_upload(file_size=file_size)

#     parts = []
#     file_hash = hashlib.sha256()
#     if self.verbose:
#         progress = tqdm(
#             desc=f"{PIPELINE_FILE_STR} Uploading {pipeline_file.path}",
#             unit="B",
#             unit_scale=True,
#             total=file_size,
#             unit_divisor=1024,
#         )
#     with open(pipeline_file.path, "rb") as f:
#         while True:
#             file_data = f.read(FILE_CHUNK_SIZE)
#             if not file_data:
#                 if self.verbose:
#                     progress.close()
#                 break
#             file_hash.update(file_data)
#             part_num = len(parts) + 1
#             # If verbose then wrap our data object in a tqdm callback
#             if self.verbose:
#                 data = CallbackBytesIO(progress.update, file_data)
#             else:
#                 data = io.BytesIO(file_data)

#             upload_metadata = self._direct_upload_pipeline_file_chunk(
#                 data=data,
#                 pipeline_file_id=pipeline_file_id,
#                 part_num=part_num,
#             )
#             parts.append(upload_metadata)

#     file_hash = file_hash.hexdigest()
#     pipeline_file_get = self._finalise_direct_pipeline_file_upload(
#         pipeline_file_id=pipeline_file_id, multipart_metadata=parts
#     )
#     # return PipelineFileVariableGet(
#     #     path=pipeline_file.path, file=pipeline_file_get.file, hash=file_hash
#     # )
