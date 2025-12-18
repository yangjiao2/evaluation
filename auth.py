#!/usr/bin/env python
# coding: utf-8

# # authentication
# ### device flow guide: https://gitlab-master.nvidia.com/kaizen/auth/starfleet/sf-docs/-/blob/main/apis/user-authentication/device-authorization-endpoint.md

# In[5]:


# !pip install nvstarfleet --extra-index-url https://urm.nvidia.com/artifactory/api/pypi/sw-cftt-pypi-local/simple
from nvstarfleet.login_user import LoginUser


# In[ ]:





# In[32]:


import requests, json

authorize_url = "https://stg.login.nvidia.com/device/authorize"
token_url = "https://stg.login.nvidia.com/token"

client_id = 'LRddW8daXcXkHW_2nvQBJt1V969xHj6m6EHf-1S4eJU' # auto (device flow)
client_id = 'rdlDOBYASPqgModybRvSuQYn_aamo_2kkdEcsQccQ9M'


device_id = '123ABC'
display_name = 'EvaluationDeviceFlow'
scope = 'openid'


# api_call_headers = {'Content-Type': 'application/x-www-form-urlencoded'}
# api_call_response = requests.post(authorize_url, headers=api_call_headers, data=data)
# response_dict = json.loads(api_call_response.text)

# input("Press Enter  after you successfully enter code on browser...")




# ### Step 1: Get Code and URL for user
# 
# 

# In[51]:


import requests

authorize_url = 'https://stg.login.nvidia.com/device/authorize'
client_id = 'LRddW8daXcXkHW_2nvQBJt1V969xHj6m6EHf-1S4eJU'
device_id = '123ABC' # random
display_name = 'deviceFlow'
scope = 'openid email profile'
# scope = 'openid'

headers = {'Content-Type': 'application/x-www-form-urlencoded'}

data = {
    'client_id': client_id,
    'device_id': device_id,
    'display_name': display_name,
    'scope': scope
}

response = requests.post(authorize_url, headers=headers, data=data)

# Print the response status code and content
print("Response Status Code:", response.status_code)

if response.status_code == 200:
    response_dict = json.loads(response.text)
    print("Go to this url", response_dict["verification_uri"])
    print("And Type this code", response_dict["user_code"])
    # After login, Page goes to: https://static-login-stg.nvidia.com/service/default/confirm?device=deviceFlow&state=x_5LhBYdHDlhvlznJi2iDyvh8Rt8Hdn9OBRHhbxAeegHtE_1flX4ehBdU9lSh-dUYvIC90EpH6lt0QnmE8HVXQ&application=NVBot&redirect_uri=https%3A%2F%2Fstg.login.nvidia.com%2Fcallback%2Fdevice_confirmation

else: 
    print (f"Invalid request, status code = {response_dict.status_code}")


# In[34]:


device_code = response_dict["device_code"]
device_code


# ##  Step 2:  get the Access Token

# In[35]:


# Step2 get the Access Token
# #curl --location --request POST 'https://stg.login.nvidia.com/token' \
# #--header 'Content-Type: application/x-www-form-urlencoded' \
# #--data-urlencode 'grant_type=urn:ietf:params:oauth:grant-type:device_code' \
# #--data-urlencode 'client_id=noOZv6QmmrKNfm7OXLsarGLiZaL1H7jBBrBs9Q7WqAM' \
# #--data-urlencode 'device_code=QjCROF2_SCWrIn5jP8aZhdlgnWtFP15euCMhH23y1EjMbPqbbRp8CKOtfPR5JaqDwvwVsiRA6ahJhaoQ8lIUIA'

token_url = "https://stg.login.nvidia.com/token"
data = {
 'device_code' : device_code,
 'grant_type' : 'urn:ietf:params:oauth:grant-type:device_code',
 'client_id' : client_id,
 'device_id': device_id,
 'scope': scope,
}
response = requests.post(authorize_url, headers=headers, data=data)

# Print the response status code and content
print("Response Status Code:", response.status_code)
response_dict = json.loads(response.text)
print (response_dict)

# api_call_headers = {'Content-Type': 'application/x-www-form-urlencoded'}
# api_call_response = requests.post(token_url, headers=api_call_headers, data=data)
# print(api_call_response.__dict__)
# response_dict = json.loads(api_call_response.text)
# print("Access Token", response_dict["access_token"])
# access_token = response_dict["access_token"]
if response.status_code == 200:
    response_dict = json.loads(response.text)
    print("Verification uri", response_dict["verification_uri_complete"])
    print("User code", response_dict["user_code"])
    print("Device code", response_dict["device_code"])
    print("Expire", response_dict["expires_in"])
    # After login, Page goes to: https://static-login-stg.nvidia.com/service/default/confirm?device=deviceFlow&state=x_5LhBYdHDlhvlznJi2iDyvh8Rt8Hdn9OBRHhbxAeegHtE_1flX4ehBdU9lSh-dUYvIC90EpH6lt0QnmE8HVXQ&application=NVBot&redirect_uri=https%3A%2F%2Fstg.login.nvidia.com%2Fcallback%2Fdevice_confirmation

else: 
    print (f"Invalid request, status code = {response_dict.status_code}")


# In[ ]:





# ### Step 3 Send Access Token to Server

# In[36]:


# api_call_headers = {'Authorization': access_token}
# local_server_url = "http://127.0.0.1:8080/auth"
# api_call_response = requests.post(local_server_url, headers=api_call_headers)
# print(api_call_response.__dict__)


# In[ ]:





# In[ ]:





# In[ ]:





# ### user flow guide: https://gitlab-master.nvidia.com/kaizen/auth/starfleet/sf-docs/-/blob/main/apis/user-authentication/README.md

# In[49]:


import requests

authorize_url = 'https://stg.login.nvidia.com/authorize'
client_id = 'rdlDOBYASPqgModybRvSuQYn_aamo_2kkdEcsQccQ9M'
redirect_uri = 'https://nvbot-stg.nvidia.com/callback'
scope = 'openid profile email'
state = 'abcd1234'

params = {
    'response_type': 'code',
    'client_id': client_id,
    'redirect_uri': redirect_uri,
    'scope': scope,
    'state': state
}

response = requests.get(authorize_url, params=params)

# Print the response status code and content
print("Response Status Code:", response.status_code)
# response_dict = json.loads(response.json())
print (response.text)
# print("Response Content:", response_dict)
# print("Go to this url", response_dict["verification_uri"])

