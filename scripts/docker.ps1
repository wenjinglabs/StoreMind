# 中文 Docker 快捷命令
#
# Docker 的服务名/容器名只允许 ASCII（[a-zA-Z0-9][a-zA-Z0-9_.-]），
# 所以底层仍然是 ollama / ollama-import / ollama-pull / app / mcp / postgres；
# 每个容器都有中文 label，这个脚本用中文别名操作。
#
# 用法：
#   .\scripts\docker.ps1 启动        # 起全部（含 ollama，必要时导入/拉模型）
#   .\scripts\docker.ps1 重新构建    # 改了代码后重建镜像再起
#   .\scripts\docker.ps1 状态
#   .\scripts\docker.ps1 日志 应用
#   .\scripts\docker.ps1 停止
#
# 中文别名：应用 / 接口 / 数据库 / 模型 / 模型导入 / 模型拉取

param(
    [Parameter(Position = 0)]
    [ValidateSet("启动", "启动模型", "重新构建", "停止", "状态", "日志", "帮助")]
    [string]$Action = "状态",

    [Parameter(Position = 1)]
    [string]$Name = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$compose = Join-Path $root "deploy\docker-compose.yml"
$docker = "C:\Program Files\Docker\Docker\resources\bin\docker.exe"

$aliases = @{
    "应用"     = "app"
    "接口"     = "mcp"
    "数据库"   = "postgres"
    "模型"     = "ollama"
    "模型导入" = "ollama-import"
    "模型拉取" = "ollama-pull"
}

function Invoke-Base([string[]]$Extra) {
    & $docker compose -f $compose @Extra
}

switch ($Action) {
    "启动" {
        Invoke-Base -Extra @("up", "-d", "--wait", "--wait-timeout", "1800")
    }
    "启动模型" {
        Invoke-Base -Extra @("up", "-d", "--wait", "--wait-timeout", "1800")
    }
    "重新构建" {
        Invoke-Base -Extra @("up", "-d", "--build", "--wait", "--wait-timeout", "1800")
    }
    "停止" {
        Invoke-Base -Extra @("down")
    }
    "状态" {
        $fmt = 'table {{.Label "com.storemind.name"}} | {{.Names}} | {{.Status}}'
        & $docker ps -a --filter "label=com.storemind.name" --format $fmt
    }
    "日志" {
        if (-not $Name) { throw "请指定中文名：应用 / 接口 / 数据库 / 模型 / 模型导入 / 模型拉取" }
        $service = $aliases[$Name]
        if (-not $service) { throw "未知服务：$Name（可用：$($aliases.Keys -join ' / ')）" }
        Invoke-Base -Extra @("logs", "-f", $service)
    }
    "帮助" {
        Write-Host "启动       起全部服务（含 ollama；模型缺失时自动拉取）"
        Write-Host "重新构建   改了代码后重建 app 镜像"
        Write-Host "状态       用中文 label 显示容器状态"
        Write-Host "日志 应用  查看某个服务的日志（应用/接口/数据库/模型）"
        Write-Host "停止       停止全部"
    }
}
