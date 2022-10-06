import base64
from typing import Union

class File():
    def __init__(self, path: str) -> None:
        self.path = path
        self._key = "ext_file" # Used to append the extension to the file string and extract it during pipeline execution
        pass

    def _file_to_str_with_extension(self, file_path: str) -> str:
        ''' 
        Convert a file to a string so that it can be sent over the API. The file extension is extracted and appended to string.

        Args:
            file_path: Path to file to be sent over the API. 
        Returns:
            File as string + extension appended
        '''
        with open(file_path, 'rb') as f:
            return base64.b64encode(f.read()).decode() + f'{self._key}{file_path[file_path.rfind("."):]}{self._key}'

    def _remove_ext_from_string(self, file_as_str_with_extension: str) -> str:
        '''
        Removes the extension from the string

        Args:
            file_as_str_with_extension: File as string with the extension appended
        Returns:
            cleaned_string: File as string without the extension
            file_ext: Extension from file
        '''
        cleaned_string = file_as_str_with_extension[:file_as_str_with_extension.find(self.key)]
        file_ext = file_as_str_with_extension[file_as_str_with_extension.find(self.key)+len(self.key):file_as_str_with_extension.rfind(self.key)]
        return cleaned_string, file_ext

    def file_str_to_b64(self, file_as_str_with_extension: str) -> Union[bytes, list]:
        cleaned_str, file_ext = self._remove_ext_from_string(file_as_str_with_extension)
        file_as_b64 = base64.decodebytes(cleaned_str.encode())
        return file_as_b64, file_ext
    
    def to_api(self) -> str:
        '''
        Provides a unified method to send io objects across the API. This should be the last stage before passing the desired object to the api.run() function.
        '''
        return self._file_to_str_with_extension(self.path)