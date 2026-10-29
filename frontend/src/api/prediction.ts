import type { PredictRequest, PredictResponse } from '../types';

export class ApiError extends Error {
  status?: number;
  details?: unknown;

  constructor(message: string, status?: number, details?: unknown) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.details = details;
  }
}

export function getApiBaseUrl(): string {
  const url = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';
  return url.replace(/\/+$/, '');
}

/**
 * Submit symptoms to the live FastAPI service. Failures are surfaced to the caller.
 */
export async function predictConditions(request: PredictRequest): Promise<PredictResponse> {
  const baseUrl = getApiBaseUrl();
  const endpoint = `${baseUrl}/api/v1/predict`;

  let response: Response;
  try {
    response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });
  } catch (err) {
    const errorMsg = err instanceof Error ? err.message : String(err);
    throw new ApiError(
      `Network error: Unable to connect to the backend server at ${baseUrl}. Please check your connection and verify the server is running. (${errorMsg})`,
    );
  }

  // Handle validation error (422)
  if (response.status === 422) {
    let errorDetail = 'Validation error: Invalid input data submitted.';
    try {
      const errJson = await response.json();
      if (errJson?.detail && Array.isArray(errJson.detail)) {
        errorDetail = errJson.detail
          .map((d: { msg?: string; loc?: (string | number)[] }) => {
            const field = d.loc ? d.loc.slice(1).join('.') : '';
            return field ? `${field}: ${d.msg}` : d.msg;
          })
          .filter(Boolean)
          .join('; ');
      } else if (typeof errJson?.detail === 'string') {
        errorDetail = errJson.detail;
      }
    } catch {
      // ignore json parse error
    }
    throw new ApiError(errorDetail, 422);
  }

  // Handle other HTTP errors
  if (!response.ok) {
    let detailMsg = `HTTP Error ${response.status}: ${response.statusText}`;
    try {
      const errBody = await response.text();
      if (errBody) {
        detailMsg = errBody;
      }
    } catch {
      // ignore
    }
    throw new ApiError(detailMsg, response.status);
  }

  let data: PredictResponse;
  try {
    data = (await response.json()) as PredictResponse;
  } catch (err) {
    throw new ApiError(
      `Failed to parse response from server: ${err instanceof Error ? err.message : String(err)}`,
      response.status,
    );
  }

  return data;
}