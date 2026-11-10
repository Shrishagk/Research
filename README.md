# **PROBLEM STATEMENT AND PROPOSED SOLUTION Compensated but Unaudited: Measuring and Closing the Drift-Compensation Blind Spot for Slow Manipulation of Low-Cost IoT Sensor Streams** 

## **1. Problem Statement** 

Low-cost Internet of Things (IoT) sensors are widely used to collect environmental and operational measurements. Their readings can change because of legitimate factors such as temperature, humidity, sensor aging, and device-to-device variation. Machine-learning (ML)-based environmental compensation can reduce these effects by estimating normal variation and correcting the measured signal. However, an adaptive compensation model may also learn to accommodate a slowly changing malicious bias. If the manipulation resembles ordinary drift, the model may partially absorb it into its correction process, leaving the output apparently stable while the underlying measurement has been altered. 

This creates a potential drift-compensation blind spot: improving measurement accuracy under changing environmental conditions does not necessarily guarantee that the system can identify malicious manipulation. A sensor stream may remain plausible after compensation even when an attacker is gradually shifting the reported values. A detector that relies only on the compensated output, or on the same adaptive model that performs the correction, may fail to distinguish genuine environmental change from adversarial manipulation. This risk must be tested experimentally rather than assumed. 

A second issue concerns device identity. Conventional software credentials and network authentication can help identify a device or protect communication, but they do not by themselves establish that the physical sensing unit is the enrolled device, nor do they prove that its measurements are truthful. A nanomaterial-based physical unclonable function (PUF) could provide a device-specific physical fingerprint. Yet its response may vary with environmental conditions, repeated measurements, and aging, so its reliability and discrimination between devices must also be evaluated. 

The research problem is therefore to determine whether adaptive environmental compensation can conceal slow sensor-stream manipulation, develop an independent integrity-monitoring mechanism to detect such manipulation, and assess whether a nanomaterial PUF can add a complementary physical-identity check without confusing device authentication with measurement integrity. 

## **2. Proposed Solution** 

The proposed solution is a layered trust architecture for low-cost IoT sensors. It combines two parallel analytical paths with a nanomaterial-based physical-identity layer. Each component has a separate responsibility: Path A corrects legitimate environmental variation, Path B independently monitors the stream for manipulation, and the PUF layer checks whether the responding device matches its enrolled physical identity. 

### **2.1 Path A — ML-Based Environmental Compensation** 

Path A estimates the expected effect of legitimate environmental conditions on sensor readings and generates a compensated measurement. Depending on the sensor and available data, candidate 

methods may include linear or tree-based regression and online/adaptive models. The model should be trained and evaluated using appropriate environmental metadata, reference measurements where available, and sensor-wise or time-wise data splits to reduce data leakage. 

Path A should output the corrected reading and, where practical, an estimate of uncertainty or a residual relative to the expected sensor response. Its purpose is measurement correction—not deciding whether the device or reading is trustworthy. 

### **2.2 Path B — Independent Integrity Monitoring** 

Path B monitors indicators that are not solely determined by the adaptive compensation output. Candidate inputs include the raw sensor stream, the compensated stream, compensation residuals, the rate and direction of model adaptation, and independent reference or peer-sensor signals when available. Candidate detectors include CUSUM, Page–Hinkley, adaptive-window drift detection, or a separately calibrated anomaly model. 

The monitor should be evaluated against gradual bias injection, abrupt manipulation, and legitimate environmental changes. Its performance should be reported using detection rate, false-alarm rate, detection delay, and minimum detectable bias slope. Comparisons should be made at matched or explicitly reported false-alarm rates. The design should avoid allowing the same adaptive model to both normalize a suspicious change and serve as the only judge of that change. 

### **2.3 Nanomaterial-Based Physical Fingerprint (PUF)** 

A suitably fabricated nanomaterial structure—for example, a random conductive nanowire network —may exhibit a device-specific electrical response that can be enrolled as a physical fingerprint. During enrollment, responses are recorded under defined conditions. During later authentication, a new response is compared with the enrolled reference using a calibrated matching rule. Environmental compensation or robust feature extraction may be investigated to improve repeatability, but it must not erase the differences that distinguish one device from another. 

The PUF is intended to support physical-device authentication, not to certify the truth of every measurement. An attacker may manipulate readings from a genuine device that passes authentication; therefore, the PUF must complement rather than replace Path B. PUF evaluation should include intra-device repeatability, inter-device separation, false acceptance and false rejection rates, environmental robustness, and—where feasible—resistance to spoofing or modeling. 

### **2.4 A detectability law with the compensator in the loop** 

Derive and validate the minimum detectable slope as a function of adaptation rate, noise, and what the monitor sees: output only, compensator state, or peer/reference data. Theory exists for known linear systems. One recent paper designs detectors against slow, crafted bias injection in stochastic linear systems and quantifies stealthiness with KL divergence. I found nobody who has done this for a learned, adaptive compensator on low-cost sensors. A forgetting-factor least-squares compensator with a closed-form analysis is enough to start. You can also frame each information source by how much it shrinks the set of stealthy attacks. A recent paper uses a similar “attack margin” for physics constraints in pedestrian sensing. 

## **3. Research Questions** 

**RQ1:** Under what conditions does adaptive environmental compensation reduce the detectability of gradual malicious bias? 

**RQ2:** Can an independent integrity monitor detect gradual manipulation while maintaining an acceptable false-alarm rate during legitimate environmental variation? 

**RQ3:** Can a nanomaterial PUF distinguish enrolled devices reliably across repeated measurements and relevant environmental conditions? 

**RQ4:** Does combining PUF-based device authentication with independent stream monitoring reduce undetected attacks compared with either mechanism alone? 

## **4. Evaluation Plan** 

**Datasets and testbed:** Begin with a suitable public sensor dataset containing environmental context and, where possible, reference measurements. Then validate on physical sensor units. Add nanomaterial PUF samples only when fabrication or collaborator-provided measurements are available. 

**Scenarios:** Normal environmental variation; sensor aging or baseline drift; gradually increasing bias; abrupt manipulation; device substitution or impersonation; and combinations of identity and stream attacks. 

**Baselines:** Static calibration with thresholds; adaptive compensation alone; compensation plus conventional drift detection; proposed independent monitor; and the full system with PUF authentication. 

**Metrics:** MAE/RMSE for compensation; detection rate, false-alarm rate, detection delay, and minimum detectable bias slope for monitoring; false acceptance/rejection and intra-/inter-device response statistics for the PUF; plus latency, memory, energy, and cost where measurable. 

**Experimental controls:** Use sensor-wise and time-wise held-out evaluation, avoid leakage between training and test data, report thresholds and attack-generation procedures, and compare detection methods at comparable false-alarm rates. 

## **5. Expected Contribution and Scope** 

The intended contribution is an empirical characterization of the trade-off between environmental compensation and slow-attack detectability, an independently evaluated integrity monitor, and a measured assessment of whether nanomaterial PUF authentication adds value as a separate physicalidentity signal. These contributions are hypotheses to validate, not established results. Novelty must be confirmed through a focused review of prior work on adaptive sensor calibration, stealthy sensor attacks, drift detection, and nanomaterial-based PUFs. 

To keep the first study feasible, validate the compensation and integrity-monitoring paths before making nanomaterial fabrication a dependency. The PUF can initially be evaluated as a separate module using suitable existing samples or collaborator-provided data, followed by integration if both components show reliable performance. 

## **6. Important Distinction** 

Device authenticity and data integrity are different properties. A PUF can support the claim that a device matches an enrolled physical identity. An integrity monitor can support the claim that a stream contains suspicious changes. Cryptography can protect messages against unauthorized alteration in transit. None of these mechanisms alone proves that a sensor's physical reading is correct; the proposed architecture combines them to address different threat surfaces. 

