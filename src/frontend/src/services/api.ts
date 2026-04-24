import axios from 'axios';

const normalizedBase = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');

export const apiClient = axios.create({
  baseURL: normalizedBase || undefined,
});

export const toApiUrl = (path: string): string => {
  if (!path) {
    return path;
  }

  if (/^https?:\/\//i.test(path)) {
    return path;
  }

  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  if (!normalizedBase) {
    return normalizedPath;
  }

  return `${normalizedBase}${normalizedPath}`;
};
