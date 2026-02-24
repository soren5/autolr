# try to create a file in the root of the project 
with open('test_file.txt', 'w') as f:
    f.write('This is just a test file to check if the file watcher is working correctly.\n')

# try to create a file in drive/bai/pfcarvalho/logs, create missing directories if necessary
import os
log_dir = 'drive/bai/pfcarvalho/logs'
os.makedirs(log_dir, exist_ok=True)
with open(os.path.join(log_dir, 'test_log.txt'), 'w') as f:
    f.write('This is just a test log file to check if the file watcher is working correctly.\n')