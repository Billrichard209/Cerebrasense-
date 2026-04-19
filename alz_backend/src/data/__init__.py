"""Dataset contracts and dataset-specific access layers."""

from .base_dataset import DatasetSample
from .external_cohort import (
    EXTERNAL_COHORT_REQUIRED_COLUMNS,
    ExternalCohortManifestError,
    ExternalCohortManifestSummary,
    build_external_cohort_records,
    load_external_cohort_manifest,
    summarize_external_cohort_manifest,
)
from .external_manifest_builder import (
    SUPPORTED_3D_IMAGE_GLOBS,
    ExternalManifestBuildError,
    ExternalManifestBuildResult,
    ExternalManifestBuilderConfig,
    build_external_cohort_manifest,
    discover_external_3d_images,
)
from .inventory import build_dataset_inventory_snapshot
from .inspectors import inspect_datasets, inspect_kaggle_dataset, inspect_oasis_dataset
from .inspection_models import DatasetInspectionReport, ImagingFileRecord
from .kaggle_alz import (
    KaggleDatasetOrganization,
    KaggleManifestResult,
    build_kaggle_manifest,
    detect_kaggle_dataset_organization,
)
from .kaggle_dataset import (
    KaggleDatasetSpec,
    build_kaggle_dataset_spec,
    build_kaggle_monai_dataloader,
    build_kaggle_monai_dataset,
    build_kaggle_monai_records,
    infer_kaggle_class_names,
    infer_kaggle_dataset_type,
    load_kaggle_manifest,
)
from .kaggle_upload_bundle import KaggleUploadBundleError, KaggleUploadBundleResult, build_kaggle_upload_bundle
from .kaggle_splits import KaggleSplitResult, build_kaggle_splits
from .loaders import (
    OASISDataloaderBundle,
    OASISDatasetBundle,
    OASISLoaderConfig,
    OASISSplitArtifacts,
    build_oasis_dataloaders,
    build_oasis_datasets,
)
from .oasis1 import OASIS1ManifestResult, build_oasis1_manifest
from .oasis1_splits import OASIS1SplitResult, build_oasis1_splits
from .oasis2 import (
    OASIS2RawInventoryResult,
    OASIS2SessionManifestResult,
    OASIS2SourceLayout,
    build_oasis2_raw_inventory,
    build_oasis2_session_manifest,
    resolve_oasis2_source_layout,
)
from .oasis2_dataset import (
    OASIS2AdapterSummary,
    OASIS2DatasetSpec,
    build_oasis2_adapter_summary,
    build_oasis2_dataset_spec,
    build_oasis2_monai_records,
    load_oasis2_session_manifest,
    save_oasis2_adapter_summary,
)
from .oasis2_metadata import (
    OASIS2MetadataAdapterSummary,
    OASIS2MetadataTemplateResult,
    build_oasis2_metadata_template,
    load_oasis2_metadata_template,
    merge_oasis2_metadata_template,
    save_oasis2_metadata_adapter_summary,
)
from .oasis2_split_policy import (
    OASIS2SplitPolicySummary,
    build_oasis2_subject_safe_split_plan,
)
from .oasis2_upload_bundle import OASIS2UploadBundleResult, build_oasis2_upload_bundle
from .oasis2_readiness import (
    OASIS2ReadinessCheck,
    OASIS2ReadinessReport,
    OASIS2_SOURCE_ENV_VAR,
    build_oasis2_readiness_report,
    resolve_oasis2_source_root,
    save_oasis2_readiness_report,
)
from .oasis_upload_bundle import OASISUploadBundleError, OASISUploadBundleResult, build_oasis_upload_bundle
from .oasis_dataset import (
    OASISDatasetSpec,
    build_oasis_dataset_spec,
    build_oasis_monai_dataloader,
    build_oasis_monai_dataset,
    build_oasis_monai_records,
    load_oasis_manifest,
)
from .registry import build_dataset_registry, build_dataset_registry_snapshot
from .scope_data_plan import build_scope_data_plan, save_scope_data_plan

__all__ = [
    "DatasetInspectionReport",
    "DatasetSample",
    "EXTERNAL_COHORT_REQUIRED_COLUMNS",
    "ExternalCohortManifestError",
    "ExternalCohortManifestSummary",
    "ExternalManifestBuildError",
    "ExternalManifestBuildResult",
    "ExternalManifestBuilderConfig",
    "ImagingFileRecord",
    "KaggleDatasetOrganization",
    "KaggleDatasetSpec",
    "KaggleManifestResult",
    "KaggleSplitResult",
    "KaggleUploadBundleError",
    "KaggleUploadBundleResult",
    "OASISDataloaderBundle",
    "OASISDatasetBundle",
    "OASIS1ManifestResult",
    "OASIS1SplitResult",
    "OASIS2ReadinessCheck",
    "OASIS2ReadinessReport",
    "OASIS2AdapterSummary",
    "OASIS2MetadataAdapterSummary",
    "OASIS2MetadataTemplateResult",
    "OASIS2RawInventoryResult",
    "OASIS2SplitPolicySummary",
    "OASIS2DatasetSpec",
    "OASIS2SessionManifestResult",
    "OASIS2SourceLayout",
    "OASIS2UploadBundleResult",
    "OASISUploadBundleError",
    "OASISUploadBundleResult",
    "OASISDatasetSpec",
    "OASISLoaderConfig",
    "OASISSplitArtifacts",
    "OASIS2_SOURCE_ENV_VAR",
    "SUPPORTED_3D_IMAGE_GLOBS",
    "build_dataset_inventory_snapshot",
    "build_external_cohort_records",
    "build_external_cohort_manifest",
    "build_dataset_registry",
    "build_dataset_registry_snapshot",
    "build_kaggle_manifest",
    "build_kaggle_monai_dataloader",
    "build_kaggle_monai_dataset",
    "build_kaggle_monai_records",
    "build_kaggle_splits",
    "inspect_datasets",
    "inspect_kaggle_dataset",
    "inspect_oasis_dataset",
    "detect_kaggle_dataset_organization",
    "infer_kaggle_class_names",
    "infer_kaggle_dataset_type",
    "build_oasis_dataloaders",
    "build_oasis1_manifest",
    "build_oasis1_splits",
    "build_oasis2_readiness_report",
    "build_oasis2_raw_inventory",
    "build_oasis2_session_manifest",
    "build_oasis2_adapter_summary",
    "build_oasis2_dataset_spec",
    "build_oasis2_metadata_template",
    "build_oasis2_monai_records",
    "build_oasis2_subject_safe_split_plan",
    "build_oasis2_upload_bundle",
    "build_oasis_upload_bundle",
    "build_kaggle_dataset_spec",
    "build_kaggle_upload_bundle",
    "build_oasis_datasets",
    "build_oasis_monai_dataloader",
    "build_oasis_monai_dataset",
    "build_oasis_monai_records",
    "build_oasis_dataset_spec",
    "load_kaggle_manifest",
    "load_external_cohort_manifest",
    "load_oasis_manifest",
    "load_oasis2_session_manifest",
    "load_oasis2_metadata_template",
    "discover_external_3d_images",
    "merge_oasis2_metadata_template",
    "resolve_oasis2_source_root",
    "resolve_oasis2_source_layout",
    "save_oasis2_adapter_summary",
    "save_oasis2_metadata_adapter_summary",
    "save_oasis2_readiness_report",
    "build_scope_data_plan",
    "save_scope_data_plan",
    "summarize_external_cohort_manifest",
]
