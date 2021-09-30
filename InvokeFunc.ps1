# Used in CI.
function Invoke-NativeCommand {
  $command = $args[0]
  $arguments = $args[1..($args.Length)]
  & $command @arguments
  if ($LastExitCode -ne 0) {
    Write-Error "Exit code $LastExitCode while running $command $arguments"
  }
}
