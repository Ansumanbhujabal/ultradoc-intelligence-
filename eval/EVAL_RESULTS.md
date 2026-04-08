# Ultradoc Intelligence -- Evaluation Results

> Compiled: 2026-04-08 10:45:57

### Sample Docs (Company Test Data)

**Run:** 2026-04-08T10:03:36.764069  |  **Docs:** 3  |  **Test Cases:** 16  |  **Mode:** hybrid

#### Q&A Performance

| Metric | Value |
|---|---|
| **Accuracy** | **100.0%** |
| Normal Q&A Accuracy | 100.0% |
| Passed / Total | 14 / 14 |
| Avg Confidence | 0.69 |
| Avg Latency | 5.2s |
| Correct Refusals | 2 / 2 |
| False Acceptances | 0 |

#### Confidence Breakdown

| Level | Count | % |
|---|---|---|
| HIGH | 9 | 64.3% |
| MEDIUM | 3 | 21.4% |
| LOW | 2 | 14.3% |

#### Extraction Performance

| Metric | Value |
|---|---|
| **Field Accuracy** | **100.0%** |
| Fields Matched | 12 / 12 |
| Avg Completeness | 82.0% |
| Avg Latency | 5.8s |

#### Latency (p50 / p95)

| Operation | p50 | p95 |
|---|---|---|
| Upload | 4.8s | 5.5s |
| Q&A | 5.4s | 7.4s |
| Extraction | 6.8s | 6.8s |

---

### Synthetic Docs (Stress Test)

**Run:** 2026-04-08T10:45:45.407238  |  **Docs:** 160  |  **Test Cases:** 960  |  **Mode:** hybrid

#### Q&A Performance

| Metric | Value |
|---|---|
| **Accuracy** | **91.9%** |
| Normal Q&A Accuracy | 87.5% |
| Passed / Total | 735 / 800 |
| Avg Confidence | 0.49 |
| Avg Latency | 2.2s |
| Correct Refusals | 315 / 320 |
| False Acceptances | 5 |

#### Confidence Breakdown

| Level | Count | % |
|---|---|---|
| HIGH | 265 | 33.1% |
| MEDIUM | 204 | 25.5% |
| LOW | 331 | 41.4% |

#### Extraction Performance

| Metric | Value |
|---|---|
| **Field Accuracy** | **66.5%** |
| Fields Matched | 1376 / 2069 |
| Avg Completeness | 95.3% |
| Avg Latency | 1.9s |

#### Latency (p50 / p95)

| Operation | p50 | p95 |
|---|---|---|
| Upload | 1.9s | 5.2s |
| Q&A | 2.5s | 3.9s |
| Extraction | 1.7s | 3.0s |

#### Issues

- **0** errors, **100** failed cases
  - `doc_001.txt`: extraction
  - `doc_002.txt`: extraction
  - `doc_003.docx`: What mode of transportation is used for this shipment?
  - `doc_003.docx`: extraction
  - `doc_004.txt`: Who is the carrier handling this shipment?
  - `doc_004.txt`: extraction
  - `doc_005.txt`: extraction
  - `doc_006.pdf`: extraction
  - `doc_007.pdf`: Who handled the delivery on behalf of the consignee?
  - `doc_007.pdf`: extraction
  - ... and 90 more

---

### Side-by-Side Comparison

| Metric | Sample | Synthetic |
|---|---|---|
| **Q&A Accuracy** | 100.0% | 91.9% |
| Normal Q&A Accuracy | 100.0% | 87.5% |
| **Extraction Accuracy** | 100.0% | 66.5% |
| Avg Completeness | 82.0% | 95.3% |
| False Acceptances | 0 | 5 |
| Avg Confidence | 0.69 | 0.49 |
| Q&A Avg Latency | 5.2s | 2.2s |
| Extraction Avg Latency | 5.8s | 1.9s |
| Correct Refusals | 2/2 | 315/320 |

### Highlights

- Sample: Perfect Q&A accuracy (100%)
- Sample: Perfect extraction field accuracy (100%)
- Sample: Zero false acceptances (guardrails hold)
- Sample: Refusal accuracy 100.0% (2/2)
- Synthetic: Refusal accuracy 98.4% (315/320)

