# 📚 Research & Domain Resources: AI for Mineral Processing & Flotation Optimization

> **Project:** Mintek SCI Grad Hackathon — AI for Mineral Processing Optimisation  
> **Student Author:** Bonga Manzini (UNISA BSc Informatics)  
> **Problem Statement:** Developing an AI soft sensor and real-time reagent dosing optimizer for iron ore reverse cationic froth flotation circuits.

---

## 1. Domain Primer: Froth Flotation Principles

Froth flotation is the primary physico-chemical separation technology used in the global mineral industry to extract valuable minerals from gangue (waste rock).

### 1.1 Why Reverse Cationic Flotation for Iron Ore?
In iron ore beneficiation, direct flotation (floating hematite/magnetite away from silica) is difficult because iron minerals are dense and present in large volumes (typically 50–65% Fe in feed), which would require immense amounts of aeration and collector reagents to float the bulk mass.

Instead, the modern industrial standard is **Reverse Cationic Flotation**:
1. **Silica (Quartz / $SiO_2$) is the mineral floated**: Cationic collectors (ether amines) bind selectively to negatively charged quartz particles, making them hydrophobic (water-repelling).
2. **Iron Ore (Hematite / $Fe_2O_3$, Magnetite / $Fe_3O_4$) is depressed**: Corn starch (depressant) is added to coat iron oxide surfaces, rendering them hydrophilic (water-attracting).
3. **Pulp Slurry Separation**:
   - **Silica Quartz** attaches to rising air bubbles and overflows as the **froth / tailings waste**.
   - **Iron Minerals** remain suspended in the pulp liquid and sink to the bottom discharge as the **high-grade concentrate**.

```
                [ Raw Ore Feed: ~56% Fe, ~15% Silica ]
                                  │
                                  ▼
                    [ Conditioning Tank ]
              ┌───────────────────┴───────────────────┐
              ▼                                       ▼
      Corn Starch Depressant              Ether Amine Collector
  (Depresses Iron Oxide surfaces)       (Hydrophobizes Silica Quartz)
              │                                       │
              └───────────────────┬───────────────────┘
                                  ▼
                 [ Flotation Column Circuit (1–7) ]
                    Air Injection ──► Rising Bubbles
                                  │
              ┌───────────────────┴───────────────────┐
              ▼                                       ▼
    [ Froth Overflow (Top) ]              [ Underflow Discharge (Bottom) ]
    Silica Quartz attached to bubbles     Depressed Iron Oxide Concentrate
    ==> Tailings Waste Stream             ==> High-Grade Iron Product (~65% Fe, <2.0% Silica)
```

---

## 2. Chemical Reagents & Operational Mechanisms

| Reagent / Parameter | Chemical Mechanism & Role | Target Industrial Range | Operational Impact of Mis-dosing |
| :--- | :--- | :--- | :--- |
| **Ether Amine (Collector)** | Cationic surfactant ($R\text{-}O\text{-}(CH_2)_3\text{-}NH_3^+$). Adsorbs onto negatively charged quartz surfaces via electrostatic attraction and hydrophobic tail clustering. | 350 – 600 $m^3/h$ (or 40–120 g/t ore) | **Under-dosing:** Silica remains in pulp $\rightarrow$ off-spec concentrate ($>2.0\% \text{ } SiO_2$).<br>**Over-dosing:** Expensive chemical waste, excessive froth formation, inadvertent iron entrapment. |
| **Corn Starch (Depressant)** | Causticized starch macromolecule rich in amylose and amylopectin. Adsorbs selectively onto hematite/iron via hydrogen bonding and complexation with iron atoms, creating a steric hydrophilic shield. | 2,000 – 4,500 $m^3/h$ (or 200–500 g/t ore) | **Under-dosing:** Iron minerals float with silica $\rightarrow$ catastrophic drop in iron recovery ($<60\%$).<br>**Over-dosing:** High slurry viscosity, hindered settling, increased pumping energy. |
| **Ore Pulp pH** | Regulated by Sodium Hydroxide ($NaOH$) or Lime ($Ca(OH)_2$). Controls zeta potential (surface electrical charge) of quartz and iron particles. | 9.8 – 10.5 (Optimum: 10.2) | **Below pH 9.0:** Quartz surface charge becomes less negative, reducing amine adsorption efficiency.<br>**Above pH 11.0:** Amine deprotonates ($R\text{-}NH_2$), losing its positive charge and collecting ability. |
| **Ore Pulp Density** | Solid-liquid mass ratio of the slurry. Controls particle suspension, collision frequency, and froth stability. | 1.60 – 1.75 $g/cm^3$ (~50–60% solids) | **Too low:** High water consumption, low residence time.<br>**Too high:** Slurry sanding, impeller wear, hindered air bubble ascent. |
| **Column Aeration (Air Flow)** | Superficial gas velocity ($J_g$) across Columns 1 to 7. Generates bubble swarm for mineral-bubble attachment. | 200 – 350 $Nm^3/h$ per column | **Too low:** Insufficient bubble surface area $\rightarrow$ poor silica removal.<br>**Too high:** Froth turbulence, entrainment of fine iron particles into tailings. |
| **Froth Pulp Level** | Column slurry level measured by float sensors and pressure transducers. Governs froth residence time and drainage. | 400 – 550 $mm$ | **High level (shallow froth):** Rapid overflow, higher recovery but lower concentrate grade.<br>**Low level (deep froth):** Enhanced drainage of entrained iron, cleaner product, but risk of losing recovery. |

---

## 3. The Core Industrial Problem: The Laboratory Assay Delay

In conventional mineral processing plants:
1. Continuous sensors (flow meters, densitometers, pH probes, air valves) log telemetry every **20 seconds**.
2. However, the true quality indicators—**% Silica Concentrate** and **% Iron Concentrate**—cannot be measured continuously by standard pipeline sensors.
3. Operators collect physical slurry grab samples from the final concentrate stream and send them to the metallurgical laboratory for X-Ray Fluorescence (XRF) or wet titration assays.
4. **The Latency:** The laboratory turnaround takes **1 to 2 hours**.
5. **The Operational Crisis:**
   - Raw feed ore grade changes dynamically every 15–30 minutes as new mining trucks unload different ore bodies into the primary crusher.
   - If feed silica surges from 12% to 22%, the plant operates blindly under outdated reagent setpoints for up to 120 minutes.
   - Result: Tonnes of off-spec ore are produced ($SiO_2 > 2.0\%$), leading to steel mill penalty fees (\$5 to \$15 per ton) or outright shipment rejection.
   - Conversely, operators often over-compensate by heavily over-dosing expensive amine collectors, wasting hundreds of thousands of dollars and polluting tailings dams.

### The AI Solution: Soft Sensing & Predictive Optimization
- **Soft Sensor (Virtual Metrology):** An AI model predicting `% Silica Concentrate` in **real time (sub-second inference)** from online sensor streams (pH, density, air flows, feed grades).
- **Prescriptive Optimizer:** A closed-loop control algorithm recommending the exact optimal **Amina Flow** and **Starch Flow** setpoints to keep silica strictly $\le 2.0\%$ at minimum chemical cost.

---

## 4. Key Academic & Industry Citations

1. **Wills, B. A., & Finch, J. A. (2015).** *Wills' Mineral Processing Technology: An Introduction to the Practical Aspects of Ore Treatment and Mineral Recovery (8th Edition).* Butterworth-Heinemann.  
   *(Foundational reference on flotation cell mechanics, cationic collector chemistry, and starch depression of iron oxides.)*
2. **Kadlec, P., Gabrys, B., & Strandt, S. (2009).** "Data-driven soft sensors in the process industry." *Computers & Chemical Engineering*, 33(4), 795–814.  
   *(Comprehensive methodology for handling industrial sensor delays, collinearity, and online model recalibration.)*
3. **de Magalhães, E. (2017).** *Quality Prediction in a Mining Process.* Kaggle Dataset. Samarco Flotation Plant, Brazil.  
   *(6-month continuous industrial telemetry dataset containing 737,453 sensor records at 20-second sampling.)*
4. **Bergh, L. G., & Yianatos, J. B. (2011).** "Current status and challenges in flotation control." *Minerals Engineering*, 24(12), 1269–1278.  
   *(Analysis of column flotation hydrodynamic stability, froth depth, and reagent feedback loops.)*
5. **Mintek Advanced Mineral Processing Division (Randburg, South Africa):**  
   - *StarCS Process Control Suite:* Advanced multivariable control for flotation cell level, aeration balance, and cyanide monitoring.  
   - *Mintek Froth Image Analysis (FrothCam / FloatStar):* Computer vision soft sensing for bubble velocity and froth texture.
6. **Le Roux, J. D., & Craig, I. K. (2016).** "Run-of-mine ore milling and flotation circuit modelling and control: A review." *IFAC-PapersOnLine*, 49(20), 40–51.  
   *(Control architectures combining metallurgical mass balance with statistical machine learning.)*

---

## 5. Mathematical Formulations

### 5.1 Metallurgical Mass Balance & Iron Recovery Rate
The metallurgical recovery rate ($R$) calculates the percentage of valuable metal (iron) successfully retained in the concentrate stream versus the initial feed:

$$R = \frac{C \cdot c}{F \cdot f} \times 100\%$$

Using the two-product metallurgical balance (where $F = C + T$ and $F \cdot f = C \cdot c + T \cdot t$):

$$R = \frac{c \cdot (f - t)}{f \cdot (c - t)} \times 100\%$$

Where:
- $f = \text{Feed Iron Grade (\% Fe)}$
- $c = \text{Concentrate Iron Grade (\% Fe)}$
- $t = \text{Tailings Iron Grade (\% Fe)}$
- $F, C, T = \text{Mass flow rates of Feed, Concentrate, and Tailings}$

### 5.2 Specific Reagent Consumption Indices
Reagent dosing efficiency is characterized by specific chemical consumption per ton of targeted mineral:

$$\text{Specific Amina Ratio} = \frac{\text{Amina Flow Rate } (m^3/h)}{\text{Ore Pulp Flow } (m^3/h) \times \text{Pulp Density } (t/m^3) \times \frac{\% \text{ Silica Feed}}{100}}$$

$$\text{Specific Starch Ratio} = \frac{\text{Starch Flow Rate } (m^3/h)}{\text{Ore Pulp Flow } (m^3/h) \times \text{Pulp Density } (t/m^3) \times \frac{\% \text{ Iron Feed}}{100}}$$

### 5.3 Optimization Objective Function
The prescriptive optimizer solves the following constrained optimization problem in real time:

$$\min_{\mathbf{u} = [Q_{\text{amina}}, Q_{\text{starch}}]} \left( C_{\text{amina}} \cdot Q_{\text{amina}} + C_{\text{starch}} \cdot Q_{\text{starch}} \right)$$

Subject to:
1. $\widehat{\% SiO_{2,\text{conc}}}(\mathbf{x}_{\text{sensors}}, \mathbf{u}) \le \text{Threshold}_{\text{silica}} \text{ (e.g., } 2.00\% \text{)}$
2. $\widehat{\% Fe_{\text{conc}}}(\mathbf{x}_{\text{sensors}}, \mathbf{u}) \ge \text{Target}_{Fe} \text{ (e.g., } 64.50\% \text{)}$
3. $Q_{\text{amina}}^{\min} \le Q_{\text{amina}} \le Q_{\text{amina}}^{\max}$
4. $Q_{\text{starch}}^{\min} \le Q_{\text{starch}} \le Q_{\text{starch}}^{\max}$

---

## 6. Student Research Insights: Informatics Meets Heavy Metallurgy

1. **The "Data Leakage" Trap in Time-Series Telemetry:**  
   When training machine learning models on high-frequency sensor streams (20-second records), applying a naive random train-test split causes adjacent records (which share nearly identical plant conditions) to leak into both train and test partitions. This produces an artificially inflated $R^2 \approx 0.85$. By implementing chronological splitting and rolling cross-validation, we test genuine predictive forecasting.
2. **Concept Drift from Ore Pit Shifting:**  
   Mineral processing plants do not operate in a stationary distribution. When mining excavators transition to a different bench in the open pit, mineral crystal hardness, liberation size, and quartz morphology change. Soft sensors must incorporate regular recalibration or online bias correction.
3. **Physical Law Consistency:**  
   Pure black-box models can occasionally predict non-physical phenomena (e.g., negative silica content or increased silica recovery when amina is doubled). By pairing tree-based ensembles with physical bounding constraints and domain feature engineering (reagent-to-feed ratios), the system produces trustworthy recommendations for control room operators.
