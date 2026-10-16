"""Lazy compatible client. The caller must have separate human spend authorization."""
import os
import re
from urllib.parse import urlsplit


def require_live_permit(manifest_hash,permit,config):
    if permit is None or permit.get('manifest_sha256')!=manifest_hash:raise PermissionError('reviewed_run_permit_required')
    for key in ('max_completions','max_attempts','max_input_tokens','max_output_tokens'):
        if type(permit.get(key)) is not int or not 0<permit[key]<=getattr(config,key):raise PermissionError('invalid_live_ceiling:'+key)
    # This release uses one immutable shared ceiling, never a looser run after restart.
    if any(permit[k]!=getattr(config,k) for k in ('max_completions','max_attempts','max_input_tokens','max_output_tokens')):
        raise PermissionError('permit_must_match_exact_frozen_ceilings')


def validate_provider(provider):
    if set(provider)-{'base_url','api_key_env'}:raise ValueError('unknown_provider_field')
    url=provider.get('base_url','');parts=urlsplit(url)
    if parts.scheme!='https' or not parts.netloc or parts.username or parts.password or parts.query or parts.fragment:raise ValueError('https_provider_url_required')
    env=provider.get('api_key_env','OPENAI_API_KEY')
    if not isinstance(env,str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',env):raise ValueError('credential_environment_name_required')
    return url,env


def live_transport(descriptor):
    url,env=validate_provider(descriptor.get('provider',{}));key=os.environ.get(env)
    if not key:raise RuntimeError('provider_credential_environment_missing')
    from openai import OpenAI,APIStatusError
    client=OpenAI(api_key=key,base_url=url,max_retries=0,timeout=180.)
    def send(request,attempt_id):
        try:
            response=client.chat.completions.with_raw_response.create(**request,extra_headers={'x-ms-client-request-id':attempt_id})
            return response.status_code,dict(response.headers),response.parse().model_dump(mode='json')
        except APIStatusError as error:
            return error.status_code,dict(error.response.headers),error.body if isinstance(error.body,dict) else {'error':{'message':'non_json_provider_error'}}
    return send
