import sys
from networksecurity.logging import logger

class NetworkSecurityException(Exception):
    def __init__(self, error_message, error_details:sys):
        super().__init__(error_message) 
        self.error_message=error_message # actuall error message
        _,_,exc_tb = error_details.exc_info() #extracts tracebacks obj from sys exception
        self.lineno=exc_tb.tb_lineno # store line number
        self.file_name=exc_tb.tb_frame.f_code.co_filename # store file name

    # this function builds detailed error nessage and shows the file name, line number and the error message
    def __str__(self):
        return "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        self.file_name, self.lineno, str(self.error_message))
    
# if __name__=="__main__":
#     try:
#         logger.logging.info("Enter the try block")
#     except Exception as e:
#         raise NetworkSecurityException(e, sys)



