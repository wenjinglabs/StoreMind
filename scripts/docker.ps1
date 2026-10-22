# 中文 Docker 快捷命令
#
# Docker 的服务名/容器名只允许 ASCII（[a-zA-Z0-9][a-zA-Z0-9_.-]），
# 所以底层仍然是 app / mcp / postgres / ollama，
# 这个脚本用中文别名 + 容器 label 给你看中文。
#
# 用法：
#   .\scripts\docker.ps1 启动           # PG + 应用 + 接口（无模型，hashing + none）
#   .\scripts\docker.ps1 启动模型       # 额外起 ollama + 拉 bge-m3 / qwen2.5:7b
#   .\scripts\docker.ps1 状态
#   .\scripts\docker.ps1 日志 应用
#   .\scripts\docker.ps1 停止
#
# 中文别名：应用 / 接口 / 数据库 / 模型 / 模型初始化

param(
    [Parameter(Position = 0)]
    [ValidateSet("启动", "启动模型", "停止", "状态", "日志", "重启", "帮助")]
    [string]$Action = "状态",

    [Parameter(Position = 1)]
    [string]$Name = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$compose = Join-Path $root "deploy\docker-compose.yml"
$composeModel = Join-Path $root "deploy\docker-compose.ollama.yml"
$docker = "C:\Program Files\Docker\Docker\resources\bin\docker.exe"

$aliases = @{
    "应用"     = "app"
    "接口"     = "mcp"
    "数据库"   = "postgres"
    "模型"     = "ollama"
    "模型初始化" = "ollama-init"
}

function Invoke-Base([string[]]$Extra) {
    & $docker compose -f $compose @Extra
}
function Invoke-WithModel([string[]]$Extra) {
    & $docker compose -f $compose -f $composeModel @Extra
}

switch ($Action) {
    "启动" {
        Invoke-Base @("up", "-d", "--build", "--wait")
    }
    "启动模型" {
        Invoke-WithModel @("up", "-d", "--build", "--wait", "--wait-timeout", "1800")
    }
    "停止" {
        Invoke-WithModel @("down")
    }
    "重启" {
        Invoke-WithModel @("up", "-d", "--force-recreate", "--wait", "--wait-timeout", "600")
    }
    "状态" {
        $fmt = 'table {{.Label "com.storemind.name"}} | {{.Names}} | {{.Status}}'
        & $docker ps -a --filter "label=com.storemind.name" --format $fmt
    }
    "日志" {
        if (-not $Name) { throw "请指定中文名：应用 / 接口 / 数据库 / 模型 / 模型初始化" }
        $service = $aliases[$Name]
        if (-not $service) { throw "未知服务：$Name（可用：$($aliases.Keys -join ' / ')）" }
        Invoke-WithModel @("logs", "-f", $service)
    }
    "帮助" {
        Write-Host "启动       PG + 应用 + 接口（hashing + none，不需要模型）"
        Write-Host "启动模型   额外起 ollama，并拉 bge-m3 / qwen2.5:7b"
        Write-Host "状态       用中文 label 显示容器状态"
        Write-Host "日志 应用  查看某个服务的日志（应用/接口/数据库/模型）"
        Write-Host "停止       停止全部"
    }
}
