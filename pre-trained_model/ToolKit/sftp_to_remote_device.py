#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pysftp
import os
import sys
import argparse

tmp_path = './'
local_lib_path = os.path.join(tmp_path, "deploy_lib.so")

# device_ip = "100.89.224.107"
# device_uname = "terry"
# device_pwd = "nvidia"
# remote_path = "/home/terry/Desktop/sftp"

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', required=True, help='Remote hostname')
    parser.add_argument('--uname', required=True, help='Remote username')
    parser.add_argument('--pwd', required=True, help='Remote password')
    parser.add_argument('--path', required=True, help='Remote file path')
    args = parser.parse_args()
    
    argv_remote_ip = args.ip
    argv_remote_uname = args.uname
    argv_remote_pwd = args.pwd
    argv_remote_path = args.path
    
    print('Remote IP address: {0}\nRemote username: {1}\nRemote password: {2}'.format(argv_remote_ip, argv_remote_uname, argv_remote_pwd))
    
    cnopts = pysftp.CnOpts()
    cnopts.hostkeys = None   
    print("Establish connection...")
    with pysftp.Connection(argv_remote_ip, username=argv_remote_uname, password=argv_remote_pwd, cnopts=cnopts) as sftp:
        with sftp.cd(argv_remote_path):
            sftp.put(local_lib_path)
    
    print("Transmission success!!!")
        
        
if __name__ == '__main__':
    main(sys.argv[1:])