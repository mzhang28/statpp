# **Design Document: osu\! StatPP System (v3.0)**

## **"Latent Skill Factorization"**

## **1\. Executive Summary**

The v3.0 architecture moves away from the deterministic "Rule-Based" calculation of v2.0 (Phase A/B) to a purely **Data-Driven Latent Variable Model**.

Instead of asking *"How much PP is this score worth?"* (a human construct), the system asks *"What Skill Vector* $\\vec{S}$ *and Difficulty Vector* $\\vec{D}$ *best explain the probability of this score occurring?"*

The system utilizes **Asymmetric Inertia** to solve the fundamental conflict of ranking systems:

* **Players** are volatile biological entities (high plasticity, protected from "bad days").  
* **Maps** are stable geological features (high inertia, eroded only by global trends).

## **2\. Mathematical Model**

### **2.1 The Prediction Function**

We model the probability of a score as the interaction between a Player's Skill and a Map's Difficulty in a multidimensional latent space (e.g., $K=32$ dimensions).

$$\\hat{y}\_{pm} \= \\sigma \\left( (\\vec{S}\_p \\cdot \\vec{D}\_m) \- b\_m \\right)$$

* $\\vec{S}\_p$: **Player Skill Vector** (Learnable, $1 \\times K$).  
* $\\vec{D}\_m$: **Map Difficulty Vector** (Learnable, $1 \\times K$).  
* $b\_m$: **Map Bias** (Base difficulty offset).  
* $\\sigma$: Sigmoid activation (Squashes result to $0.0 \- 1.0$).

### **2.2 The "Teflon & Sponge" Update Rule**

To handle "bad days" without arbitrary rules like "Top 100 only," we implement **Asymmetric Blame Assignment** directly in the backward pass.

* **Scenario A: Overperformance** ($\\text{Score} \> \\hat{y}$)  
  * **Player:** "I played better than expected." $\\rightarrow$ **Update UP.**  
  * **Map:** "This map is easier than expected." $\\rightarrow$ **Update DOWN.**  
* **Scenario B: Underperformance** ($\\text{Score} \< \\hat{y}$)  
  * **Player:** "I had a bad day." $\\rightarrow$ **Ignored (Gradient Masked).**  
  * **Map:** "This map is harder than expected." $\\rightarrow$ **Update UP.**

## **3\. The Optimizer: Inertia & Physics**

The system does not use a single learning rate. It treats Players and Maps as objects with vastly different **Physical Mass**.

### **3.1 Parameter Groups**

| Parameter Group | Learning Rate (η) | Physics Metaphor | Behavior |
| :---- | :---- | :---- | :---- |
| **Players** ($\\vec{S}$) | **High** ($\\approx 10^{-2}$) | **Feather** | Instantly reacts to new high scores. Rises rapidly. |
| **Maps** ($\\vec{D}$) | **Tiny** ($\\approx 10^{-5}$) | **Boulder** | Stationary for single users. Only moves when pushed by thousands (Mass Adjustment). |

### **3.2 The Law of Large Numbers (Global Trends)**

* **Individual Impact:** If Player X fails a map, the map vector moves by $10^{-5}$. This is negligible noise.  
* **Mass Impact:** If 10,000 players fail a map (Global Trend), the map vector moves by $10^{-5} \\times 10,000 \= 0.1$. This is a significant difficulty adjustment.

**Result:** Maps only respond to **Systemic Misratings**, not individual player variance.

## **4\. Training Loop Implementation**

The "Phase A/B" split is removed. Training is a single continuous loop (Matrix Factorization) with custom gradient logic.

### **4.1 Pseudocode (Rust/Burn Style)**

// 1\. Forward Pass  
let prediction \= sigmoid((player\_vec \* map\_vec).sum() \- map\_bias);  
let error \= observed\_score \- prediction; 

// 2\. Define Masks  
// "Teflon Mask": Only 1.0 if player overperformed (Positive Error)  
let positive\_mask \= error.greater\_elem(0.0).float(); 

// 3\. Compute Gradients Separately  
// Player Gradient: Gated by the mask.   
// "If I failed, don't learn that I'm bad. If I succeeded, learn that I'm good."  
let player\_grad \= (error \* positive\_mask) \* map\_vec;

// Map Gradient: Learns from EVERYTHING.  
// "If players fail, get harder. If players crush it, get easier."  
let map\_grad \= error \* player\_vec;

// 4\. Apply Updates with Inertia  
// Player moves fast (High LR), Map moves slow (Low LR)  
optimizer.step\_player(player\_grad, lr=0.01);  
optimizer.step\_map(map\_grad, lr=0.00001);

## **5\. The Ranking System (Post-Process)**

Since we removed score\_pp and the "Top 100" scalar sort, we need a new way to rank players globally. We use a **Supply & Demand** weighting system.

### **5.1 Step 1: Calculate "Skill Scarcity"**

After a training epoch, we analyze the distribution of the 32 latent dimensions across the entire player base.

* If **Dimension 4** has a Global Average of $0.1$ (Rare Skill), it is high value.  
* If **Dimension 1** has a Global Average of $0.9$ (Common Skill), it is low value.

$$W\_d \= \\frac{1}{\\text{Mean}(\\vec{S}\_{\\text{all}, d}) \+ \\epsilon}$$

### **5.2 Step 2: Compute Weighted Rank**

A player's Global Rank is the magnitude of their skill vector, weighted by the scarcity of their skills.

$$\\text{Rank Score} \= \\sqrt{\\sum\_{d=0}^{K} (S\_{p,d} \\cdot W\_d)^2}$$  
**Behavior:**

* **Farming Nerf:** If everyone starts farming "Jump Maps," the global average for the "Jump Dimension" rises. $W\_{jump}$ drops. Everyone's rank inflates less.  
* **Specialist Buff:** A player who masters a weird tech style (low global average) gets a massive multiplier on those dimensions.

## **6\. Data Schema Updates**

### **6.1 Conceptual Table Structure**

* **Latent Player Space (player\_vectors):**  
  * Replaces the traditional "PP" value with a dense vector (size $K=32$) representing the player's skill in the latent dimensions.  
  * Serves as the learnable embedding for the user side of the factorization model.  
* **Latent Map Space (map\_vectors):**  
  * Stores a learnable difficulty vector (size $K=32$) and a scalar bias term.  
  * **Compound Key:** Entries are unique per (MapID, ModCombination). This allows the model to learn completely different difficulty profiles for the same map when played with different mods (e.g., Double Time vs. Hard Rock).  
* **Ground Truth (scores):**  
  * Stores only the objective observation data required for training.