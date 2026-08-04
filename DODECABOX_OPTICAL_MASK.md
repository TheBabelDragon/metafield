# MetaField Dodecabox v1 — Face Optical Mask Specification

**Light body for the physical optical substrate.**  
A better way to define this is to stop thinking of it as “paint some edges black” and define it as an **optical aperture mask** on each of the 12 identical pentagonal modules.

---

## Base geometry

### Outer pentagon
- Side length: **37.103 mm**
- Same center axis as inner pentagon
- Same rotation

### Inner pentagon
- Side length: **26.714 mm**
- Offset from outer face plane: **11.56 mm**

---

## 1. Exterior face paint mask

**Purpose**
- Prevent neighboring faces from optically coupling directly
- Force photons toward intended seams

### Black perimeter band

Apply black coating inward from every outer edge:

**Width: 3.00 mm**

```
Outer side: 37.103 mm
┌───────────────────┐
│███████████████████│  3 mm black
│██             ████│
│██             ████│
│██             ████│
│███████████████████│
└───────────────────┘
```

**Resulting clear central pentagon**  
Approximate clear side length: **~31.9 mm**

---

## 2. Inner cavity face mask

**Purpose**
- Create optical void
- Absorb stray photons
- Leave only controlled edge paths

Inner pentagon: **26.714 mm** side

Paint the **entire** inner-facing pentagonal surface **matte black**.

Then create optical gates at the five edges.

---

## 3. Inner edge photon gates

Each inner pentagon edge receives a clear strip.

**Recommended v1**

Gate width: **1.50 mm**  
Centered on each inner edge.

```
26.714 mm
████████████████████
██                ██
██  ← 1.5 mm →    ██
██                ██
████████████████████
```

The gate is **not** a hole. It is:
- polished resin
- clear acrylic insert
- or optical adhesive region

Everything around it is black.

---

## 4. Corner treatment

At pentagon vertices, do **NOT** leave black paint meeting perfectly.

Leave a clear **optical node**:

**Diameter: 2.0 mm**

**Reason**
- vertices are where dodecahedron field paths converge
- five-face junctions occur there

```
edge gate
──────●──────
      ↑
 vertex optical node
```

---

## 5. Assembly seam

Between the 12 modules:

**Recommended**

| Type | Dimension |
|------|-----------|
| Structural bond | Optical resin layer **0.10–0.20 mm** |
| Intentional photon coupling seam | Clear seam **0.75 mm** (black regions stop away from the seam) |

Cross section:

```
black face
█████
     \
      \  0.75 mm clear optical channel
       \
█████
black face
```

---

## Final per-face recipe

| Feature                  | Dimension      |
|--------------------------|----------------|
| Outer pentagon side      | 37.103 mm      |
| Inner pentagon side      | 26.714 mm      |
| Face separation          | 11.56 mm       |
| Outer black border       | 3.00 mm        |
| Inner cavity coating     | 100% matte black |
| Edge optical gate width  | 1.50 mm        |
| Vertex optical node      | 2.00 mm        |
| Bond line                | 0.10–0.20 mm   |
| Optional photon seam     | 0.75 mm        |

This creates a **black optical cavity with 5 controlled transmission channels per face**.

Across 12 faces:

- **60** edge gates
- **60** vertex nodes
- **12** illumination / sensing surfaces

That gives the dodecabox a **defined optical graph** rather than an uncontrolled glowing object.

---

*See also: `PHYSICAL_FIELD_SUBSTRATE.md`, [optical-body-s3](https://github.com/TheBabelDragon/optical-body-s3)*
