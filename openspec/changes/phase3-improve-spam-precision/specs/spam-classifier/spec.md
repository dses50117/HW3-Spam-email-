# Spam Classifier Capability - Phase 3 Deltas

## MODIFIED Requirements

### Requirement: High Recall Configuration
The system SHALL provide a recommended configuration that achieves balanced precision and recall (Precision ≥ 0.90, Recall ≥ 0.93).

#### Scenario: Recommended high-recall training
- **WHEN** user trains with Phase 2 parameters: `--class-weight balanced --ngram-range 1,2 --min-df 2 --sublinear-tf --C 0.5 --eval-threshold 0.40`
- **THEN** the model SHALL achieve spam recall ≥ 0.93 on the test set

#### Scenario: Recommended balanced training (Phase 3)
- **WHEN** user trains with Phase 3 parameters: `--class-weight balanced --ngram-range 1,2 --min-df 2 --sublinear-tf --C 2.0 --eval-threshold 0.50`
- **THEN** the model SHALL achieve Precision ≥ 0.90 AND Recall ≥ 0.93 on the test set

#### Scenario: Trade-off documentation
- **WHEN** user reads the README section on recall tuning
- **THEN** they SHALL find documented trade-offs between recall and precision with example metrics

#### Scenario: Phase 3 metrics achievement
- **WHEN** training with Phase 3 recommended configuration
- **THEN** observed metrics SHALL be approximately: Accuracy ≥ 0.98, Precision ≥ 0.92, Recall ≥ 0.96, F1 ≥ 0.94

## ADDED Requirements

### Requirement: Balanced Performance Configuration
The system SHALL document a configuration that optimizes the balance between precision and recall for production use.

#### Scenario: Precision optimization
- **WHEN** user increases evaluation threshold from 0.40 to 0.50
- **THEN** precision SHALL increase while recall SHALL decrease minimally

#### Scenario: Regularization impact
- **WHEN** user increases C from 0.5 to 2.0
- **THEN** the model SHALL have less regularization and better fit training patterns

#### Scenario: Phase comparison table
- **WHEN** user reads the README
- **THEN** they SHALL find a comparison table showing metrics progression across Phase 1 (baseline), Phase 2 (high recall), and Phase 3 (balanced)

### Requirement: Parameter Selection Guidance
The system SHALL provide clear guidance on when to use different parameter configurations based on use case requirements.

#### Scenario: False negative minimization use case
- **WHEN** user needs to minimize missing spam (false negatives)
- **THEN** documentation SHALL recommend Phase 2 configuration with lower threshold

#### Scenario: False positive minimization use case
- **WHEN** user needs to minimize incorrectly flagged ham (false positives)
- **THEN** documentation SHALL recommend Phase 3 configuration with higher threshold and stronger regularization

#### Scenario: Balanced production use case
- **WHEN** user needs production-ready balanced performance
- **THEN** documentation SHALL recommend Phase 3 configuration as the default
