import os
import requests
import json
import base64


USERS_INFO = {
    "Anni": {"id": "SYJNXKBDf9GvHynan", "token": "gGsXOFnDHyVx5DsUCxX81L6deOQ-D7rdLO1QuNxze8z"},
    "Jean": {"id": "uGt3Qf4CmQH3hQh2S", "token": "uMzIQ2a0-l4bVlm8iU0S956zm3lUX2S0TR6ZX30pn5T"},
    "Ryan": {"id": "22W9xPwQyJqd9T2BY", "token": "COWZafnRALgFoB-vyyu9Hj_TYRRsdFbCHAhJnEi3nT9"},
    "Mark": {"id": "6adNW7TXoJqcDQNDC", "token": "AM9A1XO9KvUAzEznP9r7I4sftquHgyOWqR6CTA9P7cO"},
    "Derek": {"id": "JuimrQKX5w65BcWyA", "token": "qYYCStiQUlFMU7hH3SeWcCXyqhDbEN9JtPcV9WeYtAp"}, 
    "Regina": {"id": "8egMdNMumzYj6tdwh", "token": "-vC0d6wEzacRZdgXKNey1jKlHjZvBpjAyp6TxuOqZbV"},
    "Shirley": {"id": "sercW8ZDrpxeJAvWE", "token": "YH9jqRyAfvbNwN510IPR1y3JSUVzjcgEaJEKgkYchah"}, 
    "Admin":{"id": "QWMc3HkxERPHadfmD", "token": "R5tOop09MxBG6e8Z_7LUWBtw3CeJe5AflsBcnE8uRAX"}
}


# TODO: 服务改为脚本，使用execute服务执行更简洁些
def get_rocketchat_message_file_in_mongodb(env, config):
    dest_path = os.path.join(env.cache_dir, config["dest"])
    response = requests.post(
        env.controller.http_server + "/rc_mgdb_file", 
        headers={'Content-Type': 'application/json'},
        data=json.dumps(config), 
        timeout=10
    )
    if response.status_code == 200:
        base64_string = response.json()["output"]
        decoded_data = base64.b64decode(base64_string)
        with open(dest_path, "wb") as file:
            file.write(decoded_data)
        return dest_path
    else:
        print(f"Get mongodb messages error! Response status code {response.status_code}: {response.text}")
        return None


# TODO: 服务改为脚本，使用execute服务执行更简洁些
def get_rocketchat_message_content_in_mongodb(env, config):
    response = requests.post(
        env.controller.http_server + "/rc_mgdb_messages", 
        headers={'Content-Type': 'application/json'},
        data=json.dumps(config), 
        timeout=10
    )
    if response.status_code == 200:
        return response.json()["output"]
    else:
        print(f"Get mongodb messages error! Response status code {response.status_code}: {response.text}")
        return None


def get_rocketchat_group_members_with_emails(env, config):
    headers = {
        'X-Auth-Token': USERS_INFO["Admin"]["token"],
        'X-User-Id': USERS_INFO["Admin"]["id"]
    }
    server_url = f"http://{env.controller.vm_ip}:3000"
    
    # get rid
    rid_resp = requests.get(
        f"{server_url}/api/v1/groups.list",
        headers=headers
    )
    if not rid_resp.ok:
        raise Exception(f"Failed to get rid: {rid_resp.text}")
    groups = rid_resp.json().get("groups", [])
    room_id = next((g["_id"] for g in groups if g["name"] == config["room"]), None)

    # get users
    members_resp = requests.get(
        f'{server_url}/api/v1/groups.members?roomId={room_id}', 
        headers=headers
    )
    if not members_resp.ok:
        raise Exception(f"Failed to get members: {members_resp.text}")
    user_ids = [member['_id'] for member in members_resp.json().get('members', [])]

    # get emails
    result = []
    for uid in user_ids:
        user_info_resp = requests.get(
            f'{server_url}/api/v1/users.info?userId={uid}', 
            headers=headers
        )
        if not user_info_resp.ok:
            print(f"Warning: Failed to get info for user {uid}")
            continue

        user_data = user_info_resp.json().get('user', {})
        username = user_data.get('username')
        emails = user_data.get('emails', [])
        email = emails[0].get('address') if emails else None

        result.append({
            'username': username,
            'email': email
        })

    return result
