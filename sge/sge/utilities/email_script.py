import smtplib, ssl

def send_email(body):
    from resources.secrets import sender_email, receiver_email, password 

    port = 465  # For SSL
    message = """
    Subject: Experiment Complete 

    """ + body

    # Create a secure SSL context
    context = ssl.create_default_context()

    with smtplib.SMTP_SSL("smtp.dei.uc.pt", port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)