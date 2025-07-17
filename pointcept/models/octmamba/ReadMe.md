# Point Mamba: A Novel Point Cloud Backbone Based on State Space Model with Octree-Based Ordering Strategy

## Metadata

- **Authors**: Jiuming Liu, Ruiji Yu, Yian Wang, Yu Zheng, Tianchen Deng, Weicai Ye, Hesheng Wang
- **arXiv**: [2403.06467](https://arxiv.org/abs/2403.06467)
- **Code**: [GitHub Repository](https://github.com/IRMVLab/Point-Mamba)
- **Tags**: `Computer Vision`, `Point Cloud`, `State Space Models`, `3D Segmentation`, `3D Classification`

## Key Contributions

1. **Novel Architecture**:

   - First SSM-based backbone for point clouds with octree-based ordering strategy.
   - Solves the conflict between SSM's causality requirement and point clouds' irregularity.

2. **Technical Innovation**:

   - **Octree Z-Ordering**: Globally sorts points while preserving spatial proximity.
   - **Linear Complexity**: More efficient than transformer-based methods (O(n) vs O(n²)).

3. **Performance**:
   - **ModelNet40**: 93.4% classification accuracy
   - **ScanNet**: 75.7 mIoU (semantic segmentation)

## Resources

| Type        | Link                                             |
| ----------- | ------------------------------------------------ |
| Paper (PDF) | [arXiv PDF](https://arxiv.org/pdf/2403.06467)    |
| Code        | [GitHub](https://github.com/IRMVLab/Point-Mamba) |
| BibTeX      | See below                                        |

## BibTeX Citation

```bibtex
@misc{liu2024point,
  title={Point Mamba: A Novel Point Cloud Backbone Based on State Space Model with Octree-Based Ordering Strategy},
  author={Jiuming Liu and Ruiji Yu and Yian Wang and Yu Zheng and Tianchen Deng and Weicai Ye and Hesheng Wang},
  year={2024},
  eprint={2403.06467},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
