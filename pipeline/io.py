import base64


class File:
    """
    IO file descriptor. Allows to manipulate files between client and server.

    Example: (Transfer an audio file)

        '''
        On client
        '''

        from pipeline.io import File

        FILE_PATH = "files/audio.flac"

        # Convert file into a string
        input_data = File(FILE_PATH).to_api()

        run = client.run_pipeline("pipeline_ID", [input_data])

        '''
        On server
        '''
        from pipeline.io import File
        import tempfile

        file = File()
        file_as_bytes = file.file_get_bytes(input_data)
        temp_file = tempfile.NamedTemporaryFile(delete=True, mode="wb", suffix=file.extension, dir="")
        temp_file.write(file_as_bytes)

        # Do whatever to the file
        # ...
        # Close and delete file
        temp_file.close()


    """

    def __init__(self, path: str = None) -> None:
        self.path = path
        self._key = "ext_file"  # Used to append the extension to the file string and extract it during pipeline execution
        self.extension = None
        pass

    def _file_to_str_with_extension(self, file_path: str) -> str:
        """
        Convert a file to a string so that it can be sent over the API. The file extension is extracted and appended to string.

        Args:
            file_path: Path to file to be sent over the API.
        Returns:
            File as string + extension appended
        """
        with open(file_path, "rb") as f:
            return (
                base64.b64encode(f.read()).decode()
                + f'{self._key}{file_path[file_path.rfind("."):]}{self._key}'
            )

    def _remove_ext_from_string(self, file_as_str_with_extension: str) -> str:
        """
        Removes the extension from the string.

        Args:
            file_as_str_with_extension: File as string with the extension appended
        Returns:
            cleaned_string: File as string without the extension
        """
        cleaned_string = file_as_str_with_extension[
            : file_as_str_with_extension.find(self._key)
        ]
        self.extension = file_as_str_with_extension[
            file_as_str_with_extension.find(self._key)
            + len(self._key) : file_as_str_with_extension.rfind(self._key)
        ]
        return cleaned_string

    def file_get_bytes(self, file_as_str_with_extension: str) -> bytes:
        """
        Use this method to manipulate the sent file.
        The output of this function is the final form of the file and can be directly read and manipulated or written into another file.

        Args:
            file_as_str_with_extension: File in string format as sent via the API. Includes the extension of the source file.
        Return:
            file_as_bytes: Bytes representation of the file.
        """
        cleaned_str = self._remove_ext_from_string(file_as_str_with_extension)
        file_as_bytes = base64.decodebytes(cleaned_str.encode())
        return file_as_bytes

    def to_api(self) -> str:
        """
        Provides a unified method to send io objects across the API. This should be the last stage before passing the desired object to api.run() function.
        """
        return self._file_to_str_with_extension(self.path)
