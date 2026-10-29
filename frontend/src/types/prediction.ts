export type AgeGroup = 'infant' | 'child' | 'adolescent' | 'adult' | 'senior';

export type DurationUnit = 'hours' | 'days' | 'weeks' | 'months' | 'years';

export interface Duration {
  value: number;
  unit: DurationUnit;
}

export interface PredictRequest {
  free_text_symptoms?: string | null;
  structured_symptoms?: string[] | null;
  age_group?: AgeGroup | null;
  duration?: Duration | null;
}

export interface ConditionPrediction {
  condition: string;
  confidence: number;
  description?: string | null;
}

export interface RareConditionSource {
  authority: string;
  title?: string | null;
  identifier?: string | null;
  url?: string | null;
}

export interface RareConditionMatch {
  condition_id: string;
  display_name: string;
  orpha_code: string;
  icd10_code: string;
  match_score: number;
  matched_symptoms: string[];
  sources: RareConditionSource[];
  disclaimer: string;
}

export type SymptomAssertion = 'affirmed' | 'negated' | 'uncertain' | 'historical';

export interface UnderstoodSymptom {
  symptom_key: string;
  display_name: string;
  assertion: SymptomAssertion;
  temporal_context?: string | null;
  confidence: number;
  raw_mention?: string | null;
}

export interface SpellingCorrection {
  original: string;
  suggested: string;
  confidence: number;
  clarification_recommended: boolean;
}

export interface SymptomUnderstandingResult {
  status: string;
  symptoms: UnderstoodSymptom[];
  spelling_corrections: SpellingCorrection[];
  ambiguities: string[];
  clarification_needed: boolean;
  clarification_prompt?: string | null;
  engine_used: string;
  notes?: string | null;
}

export interface PredictResponse {
  status: 'success' | 'model_unavailable' | 'error' | string;
  model_version?: string | null;
  predictions: ConditionPrediction[];
  rare_condition_matches: RareConditionMatch[];
  symptom_understanding?: SymptomUnderstandingResult | null;
  message: string;
  disclaimer: string;
  metadata?: Record<string, unknown>;
}

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  model_status: 'available' | 'unavailable' | string;
}

