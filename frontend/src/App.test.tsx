/** @vitest-environment jsdom */

import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from './App';
import type { PredictResponse } from './types';

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

function responsePayload(): PredictResponse {
  return {
    status: 'success',
    model_version: '0.2.0',
    predictions: [
      { condition: 'Migraine', confidence: 0.62, description: 'Experimental pattern score.' },
    ],
    rare_condition_matches: [
      {
        condition_id: 'sample-literature-profile',
        display_name: 'Sample literature profile',
        orpha_code: 'ORPHA:123',
        icd10_code: 'R00.0',
        match_score: 0.31,
        matched_symptoms: ['headache'],
        sources: [
          { authority: 'Orphanet', title: 'Source record', url: 'https://example.org/source' },
        ],
        disclaimer: 'Literature indicator only.',
      },
    ],
    symptom_understanding: {
      status: 'success',
      symptoms: [
        { symptom_key: 'headache', display_name: 'Headache', assertion: 'affirmed', confidence: 0.96, raw_mention: 'headache' },
        { symptom_key: 'high_fever', display_name: 'Fever', assertion: 'negated', confidence: 0.9, raw_mention: 'no fever' },
        { symptom_key: 'nausea', display_name: 'Nausea', assertion: 'uncertain', confidence: 0.76, raw_mention: 'maybe nausea' },
        { symptom_key: 'cough', display_name: 'Cough', assertion: 'historical', confidence: 0.83, raw_mention: 'last month cough' },
      ],
      spelling_corrections: [],
      ambiguities: [],
      clarification_needed: false,
      clarification_prompt: null,
      engine_used: 'groq',
      notes: null,
    },
    message: 'Prediction successfully generated.',
    disclaimer: 'Educational use only. No diagnosis.',
    metadata: {},
  };
}

function clarificationPayload(): PredictResponse {
  const response = responsePayload();
  return {
    ...response,
    symptom_understanding: {
      ...response.symptom_understanding!,
      clarification_needed: true,
      clarification_prompt: 'Are you experiencing nausea now?',
      ambiguities: ['The timing of nausea is unclear.'],
    },
  };
}

function mockSuccessfulFetch(payload = responsePayload()) {
  const fetchMock = vi.fn().mockImplementation(async () =>
    new Response(JSON.stringify(payload), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    }),
  );
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

async function submitDescription(user: ReturnType<typeof userEvent.setup>) {
  await user.type(
    screen.getByRole('textbox', { name: /your description/i }),
    'Headache, no fever, maybe nausea, and a cough last month.',
  );
  await user.click(screen.getByRole('button', { name: /understand my description/i }));
}

describe('Care Atlas live exploration', () => {
  it('renders each symptom assertion returned by the API', async () => {
    mockSuccessfulFetch();
    const user = userEvent.setup();
    render(<App />);

    await submitDescription(user);

    expect(await screen.findByRole('heading', { name: 'What we understood' })).toBeTruthy();
    expect(screen.getByText('Headache')).toBeTruthy();
    expect(screen.getByText('Fever')).toBeTruthy();
    expect(screen.getByText('Nausea')).toBeTruthy();
    expect(screen.getByText('Cough')).toBeTruthy();
    expect(screen.getByRole('combobox', { name: /headache assertion, currently affirmed/i })).toBeTruthy();
    expect(screen.getByRole('combobox', { name: /fever assertion, currently denied/i })).toBeTruthy();
    expect(screen.getByRole('combobox', { name: /nausea assertion, currently uncertain/i })).toBeTruthy();
    expect(screen.getByRole('combobox', { name: /cough assertion, currently historical/i })).toBeTruthy();
  });

  it('submits the live contract and displays common and literature results separately', async () => {
    const confirmedResponse = responsePayload();
    confirmedResponse.predictions = [
      { condition: 'Confirmed input result', confidence: 0.81, description: 'Returned for confirmed structured symptoms.' },
    ];
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify(responsePayload()), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify(confirmedResponse), { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    render(<App />);

    await submitDescription(user);
    await user.selectOptions(
      screen.getByRole('combobox', { name: /nausea assertion/i }),
      'affirmed',
    );
    await user.click(screen.getByRole('button', { name: /confirm symptoms & request patterns/i }));

    expect(await screen.findByRole('heading', { name: 'Common Condition Patterns' })).toBeTruthy();
    expect(screen.getByRole('heading', { name: 'Confirmed input result' })).toBeTruthy();
    expect(screen.queryByRole('heading', { name: 'Migraine' })).toBeNull();
    expect(screen.getByRole('heading', { name: 'Rare Condition Literature Matches' })).toBeTruthy();
    expect(screen.getByRole('link', { name: /orphanet: source record/i }).getAttribute('href')).toBe('https://example.org/source');
    expect(screen.getByText(/experimental model indicators/i)).toBeTruthy();
    expect(screen.getByText(/only the reviewed symptoms labeled affirmed/i)).toBeTruthy();

    expect(fetchMock).toHaveBeenNthCalledWith(1,
      'http://localhost:8000/api/v1/predict',
      expect.objectContaining({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      }),
    );
    const requestBody = JSON.parse(fetchMock.mock.calls[0][1]?.body as string);
    expect(requestBody.free_text_symptoms).toBe('Headache, no fever, maybe nausea, and a cough last month.');
    const confirmedBody = JSON.parse(fetchMock.mock.calls[1][1]?.body as string);
    expect(confirmedBody.free_text_symptoms).toBeNull();
    expect(confirmedBody.structured_symptoms).toEqual(['headache', 'nausea']);
    expect(confirmedBody.structured_symptoms).not.toContain('high_fever');
    expect(confirmedBody.structured_symptoms).not.toContain('cough');
  });

  it('submits edits and removals as the actual confirmed prediction request', async () => {
    const fetchMock = mockSuccessfulFetch();
    const user = userEvent.setup();
    render(<App />);

    await submitDescription(user);
    await user.selectOptions(
      screen.getByRole('combobox', { name: /nausea assertion/i }),
      'affirmed',
    );
    await user.click(screen.getByRole('button', { name: 'Remove Cough' }));
    await user.click(screen.getByRole('button', { name: /confirm symptoms & request patterns/i }));

    expect(await screen.findByRole('heading', { name: 'Common Condition Patterns' })).toBeTruthy();
    const confirmedBody = JSON.parse(fetchMock.mock.calls[1][1]?.body as string);
    expect(confirmedBody.structured_symptoms).toEqual(['headache', 'nausea']);
    expect(confirmedBody.structured_symptoms).not.toContain('cough');
    expect(screen.getByText(/did not re-run groq/i)).toBeTruthy();
  });

  it('shows loading state while the live API is pending', async () => {
    let resolveResponse!: (response: Response) => void;
    const pending = new Promise<Response>((resolve) => { resolveResponse = resolve; });
    vi.stubGlobal('fetch', vi.fn().mockReturnValue(pending));
    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByRole('textbox', { name: /your description/i }), 'A headache today.');
    await user.click(screen.getByRole('button', { name: /understand my description/i }));

    expect(screen.getByRole('button', { name: /sending to care atlas/i })).toBeTruthy();
    resolveResponse(new Response(JSON.stringify(responsePayload()), { status: 200 }));
    expect(await screen.findByRole('heading', { name: 'What we understood' })).toBeTruthy();
  });

  it('shows loading state during the confirmed prediction request', async () => {
    let resolveConfirmed!: (response: Response) => void;
    const pendingConfirmed = new Promise<Response>((resolve) => { resolveConfirmed = resolve; });
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify(responsePayload()), { status: 200 }))
      .mockReturnValueOnce(pendingConfirmed);
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    render(<App />);

    await submitDescription(user);
    await user.click(screen.getByRole('button', { name: /confirm symptoms & request patterns/i }));

    expect(screen.getByRole('button', { name: /generating confirmed patterns/i })).toBeTruthy();
    resolveConfirmed(new Response(JSON.stringify(responsePayload()), { status: 200 }));
    expect(await screen.findByRole('heading', { name: 'Common Condition Patterns' })).toBeTruthy();
  });

  it('shows clarification requests and lets the user revise the narrative', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(clarificationPayload()), { status: 200 }),
    );
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    render(<App />);

    await submitDescription(user);

    expect(await screen.findByText('Are you experiencing nausea now?')).toBeTruthy();
    expect(screen.getByText('The timing of nausea is unclear.')).toBeTruthy();
    await user.click(screen.getByRole('button', { name: /revise description/i }));
    expect(screen.getByRole('textbox', { name: /your description/i })).toBeTruthy();
  });

  it('labels Groq rule-based fallback responses without changing their assertions', async () => {
    const fallback = responsePayload();
    fallback.symptom_understanding = {
      ...fallback.symptom_understanding!,
      status: 'fallback_nlp',
      engine_used: 'rule_based',
    };
    mockSuccessfulFetch(fallback);
    const user = userEvent.setup();
    render(<App />);

    await submitDescription(user);

    expect(await screen.findByText('rule based')).toBeTruthy();
    expect(screen.getByText(/understanding status: fallback nlp/i)).toBeTruthy();
    expect(screen.getByRole('combobox', { name: /fever assertion, currently denied/i })).toBeTruthy();
  });

  it('shows an explicit API error and never renders substitute predictions', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('offline')));
    const user = userEvent.setup();
    render(<App />);

    await submitDescription(user);

    await waitFor(() => expect(screen.getByRole('alert').textContent).toMatch(/offline/i));
    expect(screen.queryByText(/mock preview/i)).toBeNull();
    expect(screen.queryByRole('heading', { name: 'Common Condition Patterns' })).toBeNull();
  });

  it('shows backend HTTP failures without entering the prediction stage', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      new Response('service unavailable', { status: 503, statusText: 'Unavailable' }),
    ));
    const user = userEvent.setup();
    render(<App />);

    await submitDescription(user);

    expect(await screen.findByRole('alert')).toBeTruthy();
    expect(screen.getByRole('alert').textContent).toMatch(/503|unavailable/i);
    expect(screen.queryByRole('heading', { name: 'What we understood' })).toBeNull();
  });

  it('requires an affirmed symptom before requesting predictions', async () => {
    const fetchMock = mockSuccessfulFetch();
    const user = userEvent.setup();
    render(<App />);

    await submitDescription(user);
    for (const symptomName of ['Headache', 'Nausea']) {
      await user.selectOptions(
        screen.getByRole('combobox', { name: new RegExp(`${symptomName} assertion`, 'i') }),
        symptomName === 'Headache' ? 'negated' : 'uncertain',
      );
    }
    await user.click(screen.getByRole('button', { name: /confirm symptoms & request patterns/i }));

    expect(screen.getByRole('alert').textContent).toMatch(/at least one symptom as affirmed/i);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('validates empty input without making an API request', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole('button', { name: /understand my description/i }));

    expect(screen.getByRole('alert').textContent).toMatch(/describe at least one symptom/i);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
