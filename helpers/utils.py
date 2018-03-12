"""
.. versionadded:: 0.1
.. codeauthor:: Nekhelesh Ramananthan <krnekhelesh@skylarkdrones.com>
"""

# Copyright (C) 2016-2017 Skylark Drones

import inspect
import logging
import os
import shutil
import smtplib
import subprocess
import sys
from collections import OrderedDict
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from functools import wraps
from math import fabs
from operator import sub

logger = logging.getLogger(__name__)


def convert_ms_to_duration(total_ms):
    """
    Convert milliseconds to duration

    :param int total_ms: Total time in milliseconds

    :returns: hours, minutes and seconds
    :rtype: dict

    Output format::

        { 'hours': 13, 'minutes': 40, 'seconds': 10 }
    """
    return OrderedDict([
        ('hours', (total_ms // (1000 * 60 * 60)) % 24),
        ('minutes', (total_ms // (1000 * 60)) % 60),
        ('seconds', (total_ms // 1000) % 60)
    ])


def calculate_wind_speed(ground_speed, air_speed):
    """
    Calculate wind speed from given air speed and ground speed. Wind speed is
    essentially the difference between the air and ground speed. This function
    does not calculate the direction of the wind.

    :param list[float] ground_speed: Ground speed of aircraft
    :param list[float] air_speed: Air speed of aircraft

    :returns: Wind speed magnitude (scalar quantity)
    :rtype: float
    """
    wind_speed = list(map(sub, air_speed, ground_speed))
    wind_speed = [round(fabs(s), 2) for s in wind_speed]
    return wind_speed


def calculate_average(values_list):
    """Calculate the average of a list

    :param list values_list: List of values that need to be averaged

    :returns: average value of the list
    :rtype: float

    :raises: ZeroDivisionError
    """
    try:
        return round(float(sum(values_list) / len(values_list)), 2)
    except ZeroDivisionError:
        logger.exception("Divide by zero error. There were no values stored in "
                         "the list to calculate the average.")


def calculate_percent(value, total):
    """Calculate percentage of a number passed as an argument.

    :param int value: Actual value
    :param int total: Total value

    :returns: Percentage
    :rtype: int
    """
    return int(value / total * 100)


def get_free_space_on_disk(disk_path):
    """
    Get the free space percent of a disk

    :param str disk_path: Absolute path of disk or any file on disk
    :return: Percentage of free disk space
    :rtype: float

    Example Usage ::

        from quark.helpers.utils import get_free_space_on_disk

        free_disk_percent = get_free_space_on_disk('/home/krnekhelesh/Documents')
        print('Home directory has {} free space'.format(free_disk_percent))
    """
    disk_path = os.path.dirname(disk_path) if os.path.isfile(disk_path) else disk_path
    if os.path.isdir(disk_path):
        total, _, free = shutil.disk_usage(disk_path)
        free_disk_space_percent = calculate_percent(free, total)
        return free_disk_space_percent
    else:
        logger.error('Invalid directory path supplied to '
                     'get_free_space_on_disk(). Skipping disk space check!')
        return None


def open_file(file):
    """
    Opens file using the platform's default viewer for that file mimetype.
    For instance, if a .kml file is passed, then it will be opened
    automatically using Google Earth if it is installed in the system.

    :param str file: Absolute path of file
    """
    logger.info('Requesting OS to handle opening of {}'.format(file))

    if not os.path.isfile(file):
        logger.error('{} file cannot be opened as it cannot be found!'.format(file))
        return

    if sys.platform == 'linux':
        subprocess.call(('xdg-open', file))
    elif sys.platform == 'win32':
        try:
            os.startfile(file)
        except NotImplementedError as err_msg:
            logger.exception(str(err_msg))
    elif sys.platform == 'darwin':
        subprocess.call(('open', file))


def send_email(username, password, recipients, subject, description, attachments, emailSentFinishedSignal):
    """
    Function to send email (via Gmail) using the SMTP protocol. It takes in
    all the arguments such as from email credentials, file attachments, subject
    and body to create the email.

    .. warning::
        Services like Gmail will reject sending emails as this function
        sends email through an insecure protocol. As such, the sender's
        Gmail account settings should have allow unauthorised apps enabled!

    :param str username: email address of sender
    :param str password: password of email address of sender
    :param list recipients: recipient(s) to whom the email has to be sent
    :param str subject: one line summary of email
    :param str description: email body
    :param list attachments: file attachments
    :param pyqtSignal emailSentFinishedSignal: Signal to indicate that the
        email has been sent

    :exception: If unable to open any of the attachments or unable to send the
        email for any reason
    """
    # Create the enclosing (outer) message
    outer_msg = MIMEMultipart()
    outer_msg['Subject'] = subject
    outer_msg['To'] = ', '.join(recipients)
    outer_msg['From'] = username
    outer_msg.attach(MIMEText(description))
    outer_msg.preamble = 'You will not see this in a MIME-aware mail reader.\n'

    # Add the attachments to the message
    for file in attachments:
        try:
            with open(file, 'rb') as fp:
                msg = MIMEBase('application', "octet-stream")
                msg.set_payload(fp.read())
            encoders.encode_base64(msg)
            msg.add_header('Content-Disposition', 'attachment', filename=os.path.basename(file))
            outer_msg.attach(msg)
        except:
            logger.exception("Unable to open one of the attachments!")

    composed = outer_msg.as_string()

    # Send the email
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as s:
            s.ehlo()
            s.starttls()
            s.ehlo()
            s.login(username, password)
            s.sendmail(username, recipients, composed)
            s.close()
        logger.info("Email sent!")
        is_email_sent = True
    except:
        logger.exception("Unable to send the email!")
        is_email_sent = False

    if is_email_sent:
        emailSentFinishedSignal.emit(True, "Your bug report has been submitted successfully. Thank you for taking the "
                                           "time to report the bug. Enjoy your day!")
    else:
        emailSentFinishedSignal.emit(False, "Your report could not be sent. Please check if you are connected to the "
                                            "internet and try again!")


def generate_file(filepath, data):
    """
    A very generic function that takes a list of strings and writes that into
    the file.

    :param str filepath: Absolute path of the file to be generated
    :param data: Data to be written into the file
    :type: list(str)

    :raises IOError: If file path is invalid or if the file in question is
        opened by another application that is locking it
    """
    logger.info("Generating {} file".format(filepath))

    if os.path.exists(filepath):
        os.remove(filepath)

    try:
        with open(filepath, 'w') as data_file:
            for line in data:
                data_file.write(line)
                data_file.write('\n')
        data_file.close()
    except IOError:
        logger.critical("Unable to create {} file. Please close other applications that may be using this "
                        "file and locking it.".format(filepath), exc_info=True)


def file_check(func):
    """
    This function is meant to be used as a decorator to perform file validity
    check which includes file existence, type and value check.

    :raise ValueError: If file check fails

    Example Usage ::

        from quark.helpers.utils import file_check

        @file_check
        def get_file_info(file)
            print(os.path.basename(file)

        try:
            get_file_info('/some/invalid/path')
        except ValueError:
            print('Invalid path passed!')
    """
    @wraps(func)
    def wrapper(*args):
        for index, var_name in enumerate(inspect.getfullargspec(func)[0]):
            # self argument in a class method should be skipped
            if var_name != 'self':
                if args[index] is None or not os.path.isfile(args[index]):
                    raise ValueError('{} is not a valid file path!'.format(args[index]))
        return func(*args)
    return wrapper


def folder_check(func):
    """
    This function is meant to be used as a decorator to perform folder validity
    check which includes folder existence, type and value check.

    :raise ValueError: If folder check fails

    Example Usage ::

        from quark.helpers.utils import folder_check

        @folder_check
        def get_folder_info(folder)
            print(len(os.listdir(folder))

        try:
            get_folder_info('/some/invalid/path')
        except ValueError:
            print('Invalid path passed!')
    """
    @wraps(func)
    def wrapper(*args):
        for index, var_name in enumerate(inspect.getfullargspec(func)[0]):
            # self argument in a class method should be skipped
            if var_name != 'self':
                if args[index] is None or not os.path.isdir(args[index]):
                    raise ValueError('{} is not a valid folder path!'.format(args[index]))
        return func(*args)
    return wrapper

