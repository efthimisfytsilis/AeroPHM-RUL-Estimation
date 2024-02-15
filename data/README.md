# Data Files Overview
1. Strain Measurements for Specimen L2-03:
  * `L2-03_strains.csv`: The processed average strain for each Quasi Static (QS) test is computed across all 90 virtual Fiber Bragg Gratings (vFBGs) positioned along the stiffener feet. Each row represents a different QS test, while each column corresponds to one of the 90 vFBGs.
  * `L2-03_increment.csv`: Incremental strain measurements for specimen L2-03, showing the measurements taken along the distributed FBG ather than solely from the specified 90 vFBGs.
2.  Prognostic Features:
  * `L2_HIS.xlsx`: File containing the health indicators for the MSP specimens.
  * `specimens_hi_ga.xlsx`: The resulting fused HI for both SSPs (training) and MSPs (testing) specimens.
3. Normalization parameters:
  * `vhi1_norm_params`: The vHI<sub>1</sub> depends on the a priori knowledge of some constants that are derived from the SSP test campaigns.
