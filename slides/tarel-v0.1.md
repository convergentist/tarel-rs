
---
title: "TAReL"
sub_title: Slides
author: convergentist
---

First Slide Title
===

This is content on the first slide

<!-- end_slide -->

Limit Order Book
===

The fundamental data structure of Order-Driven markets is the *Limit Order Book*

<!-- end_slide -->

Recurrence Plot
===

![image:width:50%](figures/rp_window_0000.png)

<!-- end_slide -->

The RP Inverse
===

From Recurrence Plot (Synthetically Generated) to Limit Order Book. 

A Recurrence Plot is defined by the matrix.

# Formulas

```latex +render
\[ \sum_{n=1}^{\infty} 2^{-n} = 1 \]
```

<!-- end_slide -->

Challenges of LOB modeling with RPs
===

- Non-Stationarity: 

LOB dynamics change during "market regime shifts", making distance estimation difficult.

- Discrete vs. Continuous: 

Prices move in ticks. Standard reconstruction models assume a continuous phase space, which can lead to "ghost" price levels that don't exist on the exchange's grid.

- Uniqueness: 

Multiple LOB configurations can produce the same RP (e.g., a price move with high volume vs. a price move with low volume might result in similar Euclidean distances in the state vector).

<!-- end_slide -->

FID
===

image quality using the Fréchet Inception Distance (FID) score

<!-- end_slide -->
