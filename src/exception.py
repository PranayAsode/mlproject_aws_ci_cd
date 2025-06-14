import sys

def error_message_details(error,error_detail:sys):
    _,_,exc_tab = error_detail.exc_info()
    file_name = exc_tab.tb_frame.f_code.co_filename
    line_no = exc_tab.tb_lineno
    error_message = "Error occured in python script name [{0}] " \
    "line no [{1}] error message [{2}]".format(file_name,line_no,str(error))

    return error_message



class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message,error_detail)

    def __str__(self):
        return self.error_message
    









