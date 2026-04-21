# Fingerprint Utilities


## Supported Fingerprints

The module supports several standard fingerprint types used in cheminformatics:

| Fingerprint | Type | Description | Bits |
|-------------|------|-------------|------|
| ECFP | Structural | Extended Connectivity Fingerprint (Morgan) | Configurable |
| MACCS | Structural | MACCS Keys fingerprint | 166 |
| RDKit | Structural | RDKit topological fingerprint | 2048 |
| Gobbi2D | Pharmacophore | Gobbi pharmacophore 2D fingerprint | Variable |
| Avalon | Structural | Avalon fingerprint | Variable |

### ECFP Parameters

ECFP (Extended Connectivity Fingerprint, also known as Morgan fingerprint) fingerprints are specified as:

- Format: `"ecfp{diameter}-{nbits}"`
- Examples: `"ecfp2-1024"`, `"ecfp4-1024"`, `"ecfp6-2048"`
- Diameter: Topological diameter (internally converted to radius = diameter/2)
- Bits: Number of bits in the fingerprint (256, 512, 1024, 2048, etc.)


### Tanimoto Similarity

The module uses **Tanimoto (Jaccard) similarity** to measure molecular similarity:
- Range: 0.0 (completely dissimilar) to 1.0 (identical)
- Formula: |A ∩ B| / |A ∪ B|
- Suitable for bit vectors and sparse representations



## Function Reference


::: molrgen.evaluation.fingeprints_utils
    options:
        show_root_toc_entry: false
        heading_level: 3
