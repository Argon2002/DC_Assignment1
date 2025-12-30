param(
  [double]$LAMBD = 0.7,
  [double]$MU = 1,
  [int]$N = 50,
  [double]$MAXT = 20000,
  [Nullable[int]]$SEED = $null,  
  [Nullable[double]]$WEIBULL_SERVICE_SHAPE = $null,
  [Nullable[double]]$WEIBULL_ARRIVAL_SHAPE = $null,
  [Nullable[int]]$QUEUE_CAPACITY = $null,
  [string]$OVERFLOW_BEHAVIOUR_OPTION = $null,
  [Nullable[int]]$MAX_RETRIES = $null
)

$DS = @(1,2,5,10)


Remove-Item -ErrorAction SilentlyContinue d*.json

foreach ($d in $DS) {
  $cmd = @("python", "queue_sim.py",
           "--lambd", "$LAMBD",
           "--mu", "$MU",
           "--max-t", "$MAXT",
           "--n", "$N",
           "--d", "$d",
           "--out", "d$d.json")

  if ($null -ne $SEED) {
    $cmd += @("--seed", "$SEED")
  }

  if ($null -ne $WEIBULL_SERVICE_SHAPE) {
    $cmd += @("--weibull_shape_service", "$WEIBULL_SERVICE_SHAPE")
  }

  if ($null -ne $WEIBULL_ARRIVAL_SHAPE) {
    $cmd += @("--weibull_shape_arrival", "$WEIBULL_ARRIVAL_SHAPE")
  }



  if ($null -ne $QUEUE_CAPACITY) {
    $cmd += @("--queue_capacity", "$QUEUE_CAPACITY")
  }

  if ($null -ne $OVERFLOW_BEHAVIOUR_OPTION -and $OVERFLOW_BEHAVIOUR_OPTION -ne "") {
    $cmd += @("--overflow_behaviour_option", "$OVERFLOW_BEHAVIOUR_OPTION")
  }

  if ($null -ne $MAX_RETRIES) {
    $cmd += @("--max_retries", "$MAX_RETRIES")
  }


  Write-Host "Running d=$d -> d$d.json"
  & $cmd[0] $cmd[1..($cmd.Length-1)]
}

Write-Host "All json files generated."
Write-Host "Now run: python queue_sim.py --plot"
python queue_sim.py --lambd $LAMBD --plot
