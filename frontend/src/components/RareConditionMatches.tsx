import React from 'react';
import type { RareConditionMatch } from '../types';

interface RareConditionMatchesProps {
  matches: RareConditionMatch[];
}

export const RareConditionMatches: React.FC<RareConditionMatchesProps> = ({ matches }) => {
  if (matches.length === 0) {
    return (
      <div className="empty-substate">
        <p>No rare condition profiles matched the entered symptoms.</p>
      </div>
    );
  }

  return (
    <div className="predictions-group rare-group">
      <div className="group-header">
        <div>
          <h3 className="group-title">Rare Condition Literature Matches</h3>
          <p className="group-subtitle">
            Heuristic hallmark matching based on peer-reviewed biomedical literature (Orphanet, NIH/NORD).
            Not clinical probabilities or diagnostic likelihoods.
          </p>
        </div>
        <span className="rare-badge">Curated Literature</span>
      </div>

      <div className="cards-stack">
        {matches.map((match) => {
          const scorePercent = Math.round(match.match_score * 100);

          return (
            <article className="candidate-card rare-card" key={match.condition_id}>
              <div className="candidate-topline">
                <div className="rare-codes">
                  <span className="code-tag orpha-tag">{match.orpha_code}</span>
                  <span className="code-tag icd-tag">ICD-10: {match.icd10_code}</span>
                </div>
                <span
                  className="score-pill rare-score"
                  title="Literature-based matching indicator computed from heuristic overlap with hallmark signs. Not a clinical diagnosis probability."
                >
                  <span className="score-val">{scorePercent}%</span>
                  <span className="score-desc">literature match indicator</span>
                </span>
              </div>

              <h4 className="candidate-name">{match.display_name}</h4>

              <div className="score-bar-track" aria-hidden="true">
                <div
                  className="score-bar-fill rare-bar"
                  style={{ width: `${Math.max(4, scorePercent)}%` }}
                />
              </div>

              {match.matched_symptoms && match.matched_symptoms.length > 0 && (
                <div className="matched-block">
                  <h5>Matched Hallmark Symptoms</h5>
                  <div className="matched-tags">
                    {match.matched_symptoms.map((symptom) => (
                      <span className="matched-tag rare-matched-tag" key={symptom}>
                        {symptom.replace(/_/g, ' ')}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {match.sources && match.sources.length > 0 && (
                <div className="evidence-block">
                  <h5>Biomedical Authority References</h5>
                  <ul className="source-list">
                    {match.sources.map((source, sIdx) => (
                      <li key={`${source.authority}-${sIdx}`}>
                        {source.url ? (
                          <a
                            href={source.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="source-link"
                          >
                            <span className="source-authority">{source.authority}:</span>{' '}
                            {source.title || source.identifier || 'Reference Details'}
                            <span aria-hidden="true" className="external-glyph">
                              {' '}
                              ↗
                            </span>
                          </a>
                        ) : (
                          <span>
                            <span className="source-authority">{source.authority}:</span>{' '}
                            {source.title || source.identifier}
                          </span>
                        )}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {match.disclaimer && (
                <p className="item-disclaimer">
                  <span className="disclaimer-bullet">ℹ</span> {match.disclaimer}
                </p>
              )}
            </article>
          );
        })}
      </div>
    </div>
  );
};
