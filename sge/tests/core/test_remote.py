from resources.email_script import send_email
from keras import backend as K

def test_gpu():
    assert K.tensorflow_backend._get_available_gpus() != []

def test_email():
    send_email('This is a test email.')