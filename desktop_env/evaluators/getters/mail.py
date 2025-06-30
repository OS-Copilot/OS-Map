import os
from email import policy
from email.parser import BytesParser
from .file import get_vm_file, get_cloud_file


def get_vm_first_mail_file_parsed(env, config):
    source_dir = f"/home/user/Maildir/new"
    local_path = os.path.join(env.cache_dir, "mail_file")
    for node in env.controller.get_vm_directory_tree(source_dir)["children"]:
        if node["type"] != "file":
            continue
        get_vm_file(env, {"path": os.path.join(source_dir, node["name"]), "dest": "mail_file"})
        break
    
    with open(local_path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)
    
    body = ""
    attachments = []
    for part in msg.walk():
        content_type = part.get_content_type()
        if content_type == "text/plain":
            body = part.get_payload(decode=True).decode(part.get_content_charset(), errors="replace")
        elif part.get_content_disposition() == "attachment":
            file_data = part.get_payload(decode=True)
            file_path = os.path.join(env.cache_dir, part.get_filename())
            with open(file_path, 'wb') as output_file:
                output_file.write(file_data)
            attachments.append(file_path)
    return {
        "to": msg["TO"],
        "subject": msg["Subject"],
        "body": body,
        "attachments": attachments
    }


def get_mail_attachments(env, config):
    if "attachments" in config:
        config["attachments"] = [get_cloud_file(env, item) for item in config.pop("attachments")]
    return config