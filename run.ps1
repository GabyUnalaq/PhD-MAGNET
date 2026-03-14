param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$ScriptName,

    [Parameter(ValueFromRemainingArguments=$true)]
    $RemainingArgs
)

python -m "scripts.$ScriptName" @RemainingArgs
