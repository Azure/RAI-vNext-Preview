Param(
      [Parameter(Mandatory, HelpMessage="Please provide a valid conda env name.")]
      $EnvName,
      [Parameter(Mandatory, HelpMessage="Please provide your Azure subcription id")]
      $SubId,
      [Parameter(Mandatory, HelpMessage="Please provide your Azure resource group name")]
      $ResourceGroup,
      [Parameter(Mandatory, HelpMessage="Please provide your AzureML workspance name")]
      $Workspace
)


function Create-ConfigJson(
    $sub_id, $rg_name, $ws_name
) {
    $json_config = @{}
    $json_config["subscription_id"] = $sub_id
    $json_config["resource_group"] = $rg_name
    $json_config["workspace_name"] = $ws_name

    ConvertTo-Json $json_config | Out-File -FilePath 'config.json' -Encoding ascii
}

function Create-ComponentConfigJson(
    $version
) {
    $json_config = @{}
    $json_config["version"] = $version

    ConvertTo-Json $json_config | Out-File -FilePath 'component_config.json' -Encoding ascii
}


Write-Host "Creating conda environment '$EnvName' with python v3.8"

Create-ConfigJson $SubId $ResourceGroup $Workspace

Create-ComponentConfigJson -version 1