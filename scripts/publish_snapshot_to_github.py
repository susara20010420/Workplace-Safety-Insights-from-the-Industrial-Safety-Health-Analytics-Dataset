#!/usr/bin/env python3
"""Create/update dashboard_snapshot.json in a GitHub repository.

Use a fine-grained GitHub token with Contents read/write permission. Do not hard-code
the token in a notebook; pass it as GITHUB_TOKEN or --token.
"""
from __future__ import annotations
import argparse, base64, json, os
from pathlib import Path
import requests


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', required=True, help='owner/repository')
    p.add_argument('--file', required=True, help='Local dashboard_snapshot.json')
    p.add_argument('--target', default='dashboard/dashboard_snapshot.json', help='Path inside repository')
    p.add_argument('--branch', default='main')
    p.add_argument('--token', default=os.environ.get('GITHUB_TOKEN'))
    p.add_argument('--message', default='Update WHSAT dashboard snapshot')
    args = p.parse_args()
    if not args.token:
        raise SystemExit('Missing token. Set GITHUB_TOKEN or pass --token.')

    local = Path(args.file)
    content = base64.b64encode(local.read_bytes()).decode('ascii')
    api = f'https://api.github.com/repos/{args.repo}/contents/{args.target}'
    headers = {
        'Authorization': f'Bearer {args.token}',
        'Accept': 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28',
    }
    params = {'ref': args.branch}
    existing = requests.get(api, headers=headers, params=params, timeout=30)
    sha = existing.json().get('sha') if existing.status_code == 200 else None
    if existing.status_code not in (200, 404):
        raise RuntimeError(f'GitHub lookup failed: {existing.status_code} {existing.text}')

    payload = {
        'message': args.message,
        'content': content,
        'branch': args.branch,
    }
    if sha:
        payload['sha'] = sha
    r = requests.put(api, headers=headers, json=payload, timeout=30)
    if r.status_code not in (200, 201):
        raise RuntimeError(f'GitHub update failed: {r.status_code} {r.text}')
    data = r.json()
    print('Published:', data.get('content', {}).get('html_url', args.target))

if __name__ == '__main__':
    main()
