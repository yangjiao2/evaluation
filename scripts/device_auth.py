#%% md
# # authentication flow
# 
# video guide: [sharepoint link](https://nvidia-my.sharepoint.com/:v:/r/personal/yangj_nvidia_com/Documents/device_auth.mov?csf=1&web=1&e=hsTcLg&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifX0%3D)
# 
# 1. login device flow
# 2. uri complete to authorize
# 3. get client_token
# 4. insert client_token to with renew TTL to redis
# 
# 
# device flow guide: https://gitlab-master.nvidia.com/kaizen/auth/starfleet/sf-docs/-/blob/main/apis/user-authentication/device-authorization-endpoint.md
#%%
import os

# Need input
CLIENT_IDS = {
    "nonprd": "LRddW8daXcXkHW_2nvQBJt1V969xHj6m6EHf-1S4eJU",
    "prd": "QHWXxZpiDCFZ8T9zCbX9QJUOotq8c6HGVlyYIvDGxXk"
}


# Need input
# user_id will be user for individual person, or could user unique bot id (in future when bot configuration is ready).

userid = ""

#%%
# Need input
# option: "stg", "prd"
env = "nonprd"
print(f"✋Verify that we will retrieve for *{env}* env")
# intentional env overwrite
env = "prd" if env.lower() == "prd" else "nonprd"
#%%
import requests, json

client_id = CLIENT_IDS.get(env)
assert client_id is not None

authorize_url_stg = 'https://stg.login.nvidia.com/device/authorize'
authorize_url_prd = 'https://login.nvidia.com/device/authorize'

authorize_url = authorize_url_stg if env.lower() == "nonprd" else authorize_url_prd

token_url_stg = "https://stg.login.nvidia.com/token"
token_url_prd = "https://login.nvidia.com/token"
token_url = token_url_stg if env.lower() == "nonprd" else token_url_prd

client_token_url_stg = "https://stg.login.nvidia.com/client_token"
client_token_url_prd = "https://login.nvidia.com/client_token"
client_token_url = client_token_url_stg if env.lower() == "nonprd" else client_token_url_prd


#%%
device_id = 'thisisadeviceidfornvbotevaluation'
display_name = f"{userid or ''}DeviceFlow"
scope = 'openid profile email'
values_dict = {"device_id": device_id, "client_id": client_id}

#%% md
# ### Step 1: get Device Code and Authorize
# 
# 
#%%
print("-> authorize_url: ", authorize_url)
print("-> client_id: ", client_id)
#%%
data = {
    'client_id': client_id,
    'device_id': device_id,
    'display_name': display_name,
    'scope': scope
}
#%%
headers = {'Content-Type': 'application/x-www-form-urlencoded'}

response = requests.post(authorize_url, headers=headers, data=data)
response.text
#%%

response_dict = None
if response.status_code == 200:
    response_dict = json.loads(response.text)
    # print("And Type this code", response_dict["user_code"])

else:
    print(f"Invalid request, status code = {response.status_code}, {response.text}")

if response_dict:
    device_code = response_dict["device_code"]
    # print ("device_code: \n", device_code)

print("\nGo to the url endpoint in next code execution block for UI login \n")
print("1. click `Submit` ")
print("2. click `Continue`")
print("3. should see `You are logged in`")
#%%
import webbrowser

webbrowser.open_new_tab(response_dict["verification_uri_complete"])
print("✋Verify complete this steps and able to see `You are logged in`")
#%% md
# ###  Step 2: get Access Token
#%%
# URL
print("-> token_url: ", token_url)
#%%
print("-> device_code: ", device_code)
#%%
data = {
    'device_code': device_code,
    'grant_type': 'urn:ietf:params:oauth:grant-type:device_code',
    'client_id': client_id,
}

token_response = requests.post(token_url, headers=headers, data=data)
print (token_response)
if token_response.status_code == 200:
    token_response_dict = token_response.json()
    print("Id token: \n", token_response_dict["id_token"])
    values_dict["id_token"] = token_response_dict["id_token"]
    access_token = token_response_dict["access_token"]
    values_dict["access_token"] = access_token
    print("Access_token: \n", token_response_dict["access_token"])
    print("Expire:\n", token_response_dict["expires_in"])
else:
    print(f"Invalid request, status code = {response_dict.status_code}")

#%%
token_response
#%%
assert token_response_dict is not None
id_token = token_response_dict['id_token']
# print (f"ID token: {id_token}")

#%%
access_token = token_response_dict['access_token']
print(f"Access token: {token_response_dict['access_token']}")

#%% md
# ## Step 3:  get client token by access token
# 
# https://gitlab-master.nvidia.com/kaizen/auth/starfleet/sf-docs/-/blob/main/overview/extras/client-tokens.md
#%%
# URL
print("-> client_token_url: ", client_token_url)
#%%

headers = {
    # 'Content-Type': 'application/json',
    'Authorization': f"Bearer {access_token}"
}

client_token_response = requests.get(client_token_url, headers=headers)

# Print the response status code and content
print("Response Status Code:", client_token_response.status_code)
print(client_token_response.text)
if client_token_response.status_code == 200:
    client_token_response_dict = json.loads(client_token_response.text)
    

#%%
assert client_token_response_dict is not None
client_token = client_token_response_dict['client_token']
values_dict["client_token"] = client_token_response_dict["client_token"]
print (f"Client token: {client_token}")
values_dict["client_token_expire_in"] = client_token_response_dict["expires_in"]
#%%
import jwt

decoded_token = jwt.decode(id_token, options={"verify_signature": False})

sub_claim = decoded_token.get("sub")
print("Sub Claim:", sub_claim)
#%%
from datetime import timedelta

def convert_seconds(seconds):
    duration = timedelta(seconds=seconds)
    days = duration.days

    # Calculate total minutes from the duration
    # Note: timedelta.total_seconds() returns the total number of seconds contained in the duration
    total_minutes = duration.total_seconds() / 60

    return days, total_minutes


print(
    f"Access token `{access_token}` expires in {convert_seconds(client_token_response_dict['expires_in'])[0]} days")


#%% md
# ### Step 4: Obtain device auth token from client token
#%%
print (env)
print (token_url)
print (client_id)
#%%
headers = {'Content-Type': 'application/x-www-form-urlencoded'}

data = {
    "client_token": client_token,
    "grant_type": "urn:ietf:params:oauth:grant-type:client_token",
    'client_id': client_id,
    'sub': sub_claim,
}
device_auth_token_response = requests.post(token_url, headers=headers, data=data)

# Print the response status code and content
print("Response Status Code:", device_auth_token_response.status_code)
print (device_auth_token_response.text)
if device_auth_token_response.status_code == 200:
    device_auth_token_response_dict = json.loads(device_auth_token_response.text)
    print ("id_token: ", device_auth_token_response_dict['id_token'])
    device_token = device_auth_token_response_dict['id_token']
    print (device_auth_token_response_dict)
#%%
{
    "client_token": client_token,
    "grant_type": "urn:ietf:params:oauth:grant-type:client_token",
    'client_id': client_id,
    'sub': sub_claim,
}
#%%

#%%
assert device_auth_token_response is not None
device_access_token = device_auth_token_response_dict['access_token']
device_token = device_auth_token_response_dict['id_token']
device_client_token = device_auth_token_response_dict['client_token']
device_auth_expires_in = device_auth_token_response_dict['expires_in']

print(f"Device token `{device_token}` expires in {convert_seconds(device_auth_expires_in)[-1]} mins")

values_dict["device_auth_token"] = device_token
values_dict["device_client_token"] = device_client_token
values_dict["device_auth_token_expire_in"] = device_auth_expires_in
#%%
import requests

service_url = "https://stgbot-api.nvidia.com/services/" if env.lower() == "nonprd" else "https://nvbot-api.nvidia.com/services/"
headers = {
    'accept': 'application/json',
    'device-initiated': 'false',
    'Authorization': f'Bearer {device_token}'
}

auth_verify_response = requests.get(service_url, headers=headers)

assert auth_verify_response.status_code == 200
print("Verified device token successfully.")

#%%
print("Verified device token successfully.")
#%%
print (f'{values_dict["device_auth_token"]} will expire in {values_dict["device_auth_token_expire_in"] }')

#%%

#%%
print (f" -- {client_id} Summary: -- \n{values_dict}")

#%%
values_dict['client_token']
#%%
sub_claim
#%%
import ipywidgets as widgets
from IPython.display import display, Javascript

def copy_to_clipboard(text):
    # Create a button to trigger the copy action
    copy_button = widgets.Button(description=f"Copy device token")
    
    def on_button_click(b):
        # JavaScript code to copy text to clipboard
        js_code = f"""
        var text = `{text}`;
        var temp_input = document.createElement('input');
        document.body.appendChild(temp_input);
        temp_input.value = text;
        temp_input.select();
        document.execCommand('copy');
        document.body.removeChild(temp_input);
        alert('Copied to clipboard: ' + text);
        """
        display(Javascript(js_code))

    # Bind the button click event to the function
    copy_button.on_click(on_button_click)
    
    # Display the button
    display(copy_button)

copy_to_clipboard(device_token)
#%% md
# ### Complete