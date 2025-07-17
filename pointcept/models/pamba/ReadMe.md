# Pamba: Enhancing Global Interaction in Point Clouds via State Space Model

## Metadata

**Authors**: Zhuoyuan Li, Yubo Ai, Jiahao Lu, et al.  
**Conference**: AAAI 2025 (Accepted)  
**arXiv**: [2406.17442v3](https://arxiv.org/abs/2406.17442v3)  
**DOI**: [10.48550/arXiv.2406.17442](https://doi.org/10.48550/arXiv.2406.17442)  
**Tags**: `3D Segmentation`, `State Space Models`, `Linear Complexity`, `Global Modeling`  
**Implementation Note**:

- Bidirectional Mamba implementation adapted from [Caduceus](#reference-1)
- Current implementation shows suboptimal performance (under optimization)

## Key Innovations

1. **Architecture**:

   - First Mamba-based model for point clouds with **multi-path serialization** strategy
   - **ConvMamba Block**: Combines SSM's global modeling with CNN's local geometry learning
   - Bidirectional processing via weight-shared Mamba blocks (following Caduceus)

2. **Technical Breakthroughs**:

   - Solves disorder-to-causal mismatch via adaptive point cloud serialization
   - **Linear complexity** (O(n)) vs transformers' quadratic cost (O(n²))
   - Bi-directional modeling without parameter doubling

3. **Performance**:
   - SOTA on **4 major benchmarks**:
     - ScanNet v2 / ScanNet200
     - S3DIS / nuScenes
   - 1.8× faster inference than transformer baselines

## References

1. **Caduceus**:  
   Schiff et al. "Bi-Directional Equivariant Long-Range DNA Sequence Modeling"  
   ICML 2024. [arXiv:2403.03234](https://arxiv.org/abs/2403.03234)  
   _Key features adopted_:
   - Weight-sharing in bidirectional SSM blocks
   - Direction-agnostic parameterization

## BibTeX Citation

```bibtex
@article{li2024pamba,
  title={Pamba: Enhancing Global Interaction in Point Clouds via State Space Model},
  author={Li, Zhuoyuan and Ai, Yubo and Lu, Jiahao and Wang, ChuXin and Deng, Jiacheng and Chang, Hanzhi and Liang, Yanzhe and Yang, Wenfei and Zhang, Shifeng and Zhang, Tianzhu},
  journal={AAAI Conference on Artificial Intelligence},
  year={2025},
  note={Accepted}
}

@article{schiff2024caduceus,
  title={Caduceus: Bi-Directional Equivariant Long-Range DNA Sequence Modeling},
  author={Schiff, Yair and Kao, Chia-Hsiang and Gokaslan, Aaron and Dao, Tri and Gu, Albert and Kuleshov, Volodymyr},
  journal={International Conference on Machine Learning},
  year={2024}
}
```
