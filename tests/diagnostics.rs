use brep_kernel::boolean::BooleanError;
use brep_kernel::errors::{
    DiagnosticSeverity, KernelDiagnostic, KernelDiagnosticReport, KernelError, KernelErrorKind,
    KernelSubsystem,
};

#[test]
fn boolean_unsupported_errors_preserve_structured_context() {
    let error = KernelError::from(BooleanError::Unsupported)
        .with_operation("build_closed_boolean_output")
        .with_note("input kept regions did not close into a shell");

    assert_eq!(error.primary.subsystem, KernelSubsystem::Boolean);
    assert_eq!(error.primary.kind, KernelErrorKind::Unsupported);
    assert_eq!(error.primary.code, "boolean.unsupported");
    assert_eq!(
        error.primary.operation.as_deref(),
        Some("build_closed_boolean_output")
    );
    assert!(error
        .primary
        .notes
        .iter()
        .any(|note| note.contains("structured diagnostics")));
    assert!(error
        .primary
        .notes
        .iter()
        .any(|note| note.contains("input kept regions")));
}

#[test]
fn diagnostic_report_promotes_first_error_and_retains_related_notes() {
    let warning = KernelDiagnostic::new(
        DiagnosticSeverity::Warning,
        KernelSubsystem::Intersection,
        KernelErrorKind::Numerical,
        "ssi.high_residual",
        "intersection residual exceeded the preferred tolerance",
    )
    .with_operation("surface_surface_intersection")
    .with_note("curve was still retained for downstream trimming");

    let error = KernelDiagnostic::new(
        DiagnosticSeverity::Error,
        KernelSubsystem::Boolean,
        KernelErrorKind::Topology,
        "boolean.open_shell",
        "classified kept regions did not form a closed shell",
    )
    .with_operation("build_closed_boolean_output");

    let report = KernelDiagnosticReport::new()
        .with_diagnostic(warning.clone())
        .with_diagnostic(error.clone());

    assert_eq!(report.warning_count(), 1);
    assert_eq!(report.error_count(), 1);

    let result = report.into_result(());
    let kernel_error = result.unwrap_err();
    assert_eq!(kernel_error.primary.code, error.code);
    assert_eq!(kernel_error.related, vec![warning]);
}
