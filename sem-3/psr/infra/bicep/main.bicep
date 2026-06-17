@description('Azure region')
param location string = resourceGroup().location

@description('Application name prefix')
param appName string = 'book-ai-library'

@description('Container image tag to deploy for all app images')
param imageTag string = 'latest'

@description('Container Apps workload profile location suffix is handled by ACA environment')
param containerCpu string = '0.25'

@description('Container Apps memory per service')
param containerMemory string = '0.5Gi'

@description('Deploy Container Apps after images have been pushed to ACR')
param deployApps bool = false

@description('Deploy Azure Database for PostgreSQL Flexible Server. Keep false for cheap infrastructure what-if checks.')
param deployPostgres bool = false

@description('Deploy a small internal Ollama Container App for POC model inference. This is not production-hardened.')
param deployOllama bool = false

@description('Ollama generation model to auto-pull when deployOllama=true')
param ollamaGenerateModel string = 'qwen3:0.6b'

@description('Ollama embedding model to auto-pull when deployOllama=true')
param ollamaEmbedModel string = 'embeddinggemma'

@description('CPU for optional Ollama Container App')
param ollamaCpu string = '2'

@description('Memory for optional Ollama Container App')
param ollamaMemory string = '4Gi'

@description('PostgreSQL admin user for Flexible Server')
param postgresAdminUser string = 'bookadmin'

@secure()
@description('PostgreSQL admin password. Required only when deployPostgres=true.')
param postgresAdminPassword string = ''

@description('Application database name')
param postgresDatabaseName string = 'book_ai_library'

@description('Azure OpenAI endpoint, for example https://name.openai.azure.com')
param azureOpenAIEndpoint string = ''

@secure()
@description('Azure OpenAI API key injected only for app deployment')
param azureOpenAIApiKey string = ''

@description('Azure OpenAI embeddings deployment name')
param azureOpenAIEmbedDeployment string = ''

@description('Azure OpenAI chat deployment name')
param azureOpenAIChatDeployment string = ''

@description('Optional external LLM Service URL. Use this for hybrid demos where Azure services call a local/remote llm-service exposed through a secure tunnel.')
param externalLlmServiceUrl string = ''

@description('Timeout in seconds for Open Library requests. Azure egress is slower than localhost, so keep this higher than local browser expectations.')
param openLibraryTimeoutSeconds string = '25'

var normalizedAppName = toLower(appName)
var nameSuffix = uniqueString(subscription().id, resourceGroup().id, normalizedAppName)
var acrName = take('bookai${replace('${normalizedAppName}${nameSuffix}acr', '-', '')}', 50)
var serviceBusName = take('${normalizedAppName}-${nameSuffix}-bus', 50)
var postgresServerName = take('${normalizedAppName}-${nameSuffix}-pg', 63)
var useAzureOpenAI = !empty(azureOpenAIApiKey) && !empty(azureOpenAIEndpoint) && !empty(azureOpenAIEmbedDeployment) && !empty(azureOpenAIChatDeployment)
var useAzureOllama = deployOllama && !useAzureOpenAI && empty(externalLlmServiceUrl)
var databaseUrl = deployPostgres ? 'postgresql://${postgresAdminUser}:${postgresAdminPassword}@${postgresServerName}.postgres.database.azure.com:5432/${postgresDatabaseName}?sslmode=require' : ''
var serviceBusConnectionString = listKeys(resourceId('Microsoft.ServiceBus/namespaces/authorizationRules', serviceBus.name, 'RootManageSharedAccessKey'), '2024-01-01').primaryConnectionString
var internalBaseDomain = acaEnv.properties.defaultDomain
var llmServiceUrl = !empty(externalLlmServiceUrl) ? externalLlmServiceUrl : 'https://${normalizedAppName}-llm-service.internal.${internalBaseDomain}'
var ollamaBaseUrl = 'https://${normalizedAppName}-ollama.internal.${internalBaseDomain}'

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: '${normalizedAppName}-logs'
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: acrName
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: true
  }
}

resource acaEnv 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: '${normalizedAppName}-env'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

resource serviceBus 'Microsoft.ServiceBus/namespaces@2024-01-01' = {
  name: serviceBusName
  location: location
  sku: {
    name: 'Standard'
    tier: 'Standard'
  }
}

resource booksTopic 'Microsoft.ServiceBus/namespaces/topics@2024-01-01' = {
  parent: serviceBus
  name: 'books'
}

resource usersTopic 'Microsoft.ServiceBus/namespaces/topics@2024-01-01' = {
  parent: serviceBus
  name: 'users'
}

resource embeddingSubscription 'Microsoft.ServiceBus/namespaces/topics/subscriptions@2024-01-01' = {
  parent: booksTopic
  name: 'embedding-worker'
}

resource recommendationBooksSubscription 'Microsoft.ServiceBus/namespaces/topics/subscriptions@2024-01-01' = {
  parent: booksTopic
  name: 'recommendation-service'
}

resource recommendationUsersSubscription 'Microsoft.ServiceBus/namespaces/topics/subscriptions@2024-01-01' = {
  parent: usersTopic
  name: 'recommendation-service'
}

resource postgresServer 'Microsoft.DBforPostgreSQL/flexibleServers@2023-12-01-preview' = if (deployPostgres) {
  name: postgresServerName
  location: location
  sku: {
    name: 'Standard_B1ms'
    tier: 'Burstable'
  }
  properties: {
    administratorLogin: postgresAdminUser
    administratorLoginPassword: postgresAdminPassword
    version: '16'
    storage: {
      storageSizeGB: 32
    }
    backup: {
      backupRetentionDays: 7
      geoRedundantBackup: 'Disabled'
    }
    highAvailability: {
      mode: 'Disabled'
    }
    network: {
      publicNetworkAccess: 'Enabled'
    }
  }
}

resource postgresDatabase 'Microsoft.DBforPostgreSQL/flexibleServers/databases@2023-12-01-preview' = if (deployPostgres) {
  parent: postgresServer
  name: postgresDatabaseName
  properties: {
    charset: 'UTF8'
    collation: 'en_US.utf8'
  }
}

resource postgresVectorExtensionAllowList 'Microsoft.DBforPostgreSQL/flexibleServers/configurations@2023-12-01-preview' = if (deployPostgres) {
  parent: postgresServer
  name: 'azure.extensions'
  properties: {
    value: 'VECTOR'
    source: 'user-override'
  }
}

resource postgresAllowAzureServices 'Microsoft.DBforPostgreSQL/flexibleServers/firewallRules@2023-12-01-preview' = if (deployPostgres) {
  parent: postgresServer
  name: 'AllowAzureServices'
  properties: {
    startIpAddress: '0.0.0.0'
    endIpAddress: '0.0.0.0'
  }
}

var services = [
  {
    name: 'frontend'
    port: 8501
    external: true
  }
  {
    name: 'llm-service'
    port: 8000
    external: false
  }
  {
    name: 'book-catalog'
    port: 8000
    external: false
  }
  {
    name: 'user-profile'
    port: 8000
    external: false
  }
  {
    name: 'embedding-worker'
    port: 8000
    external: false
  }
  {
    name: 'recommendation'
    port: 8000
    external: true
  }
]

var acrCredentials = acr.listCredentials()

resource containerApps 'Microsoft.App/containerApps@2024-03-01' = [for service in services: if (deployApps) {
  name: '${normalizedAppName}-${service.name}'
  location: location
  properties: {
    managedEnvironmentId: acaEnv.id
    configuration: {
      ingress: {
        external: service.external
        targetPort: service.port
        transport: 'auto'
      }
      registries: [
        {
          server: acr.properties.loginServer
          username: acrCredentials.username
          passwordSecretRef: 'acr-password'
        }
      ]
      secrets: concat([
        {
          name: 'acr-password'
          value: acrCredentials.passwords[0].value
        }
        {
          name: 'service-bus-connection-string'
          value: serviceBusConnectionString
        }
      ], deployPostgres ? [
        {
          name: 'database-url'
          value: databaseUrl
        }
      ] : [], useAzureOpenAI ? [
        {
          name: 'azure-openai-api-key'
          value: azureOpenAIApiKey
        }
      ] : [])
    }
    template: {
      containers: [
        {
          name: service.name
          image: '${acr.properties.loginServer}/${service.name}:${imageTag}'
          env: concat([
            {
              name: 'APP_STATE_FILE'
              value: '/tmp/app_state.json'
            }
            {
              name: 'LLM_SERVICE_URL'
              value: llmServiceUrl
            }
            {
              name: 'BOOK_CATALOG_URL'
              value: 'https://${normalizedAppName}-book-catalog.internal.${internalBaseDomain}'
            }
            {
              name: 'USER_PROFILE_URL'
              value: 'https://${normalizedAppName}-user-profile.internal.${internalBaseDomain}'
            }
            {
              name: 'RECOMMENDATION_URL'
              value: 'https://${normalizedAppName}-recommendation.internal.${internalBaseDomain}'
            }
            {
              name: 'EMBEDDING_WORKER_URL'
              value: 'https://${normalizedAppName}-embedding-worker.internal.${internalBaseDomain}'
            }
            {
              name: 'LLM_PROVIDER'
              value: service.name == 'llm-service' && useAzureOpenAI ? 'azure-openai' : (service.name == 'llm-service' && useAzureOllama ? 'ollama-with-fallback' : 'deterministic')
            }
            {
              name: 'OLLAMA_BASE_URL'
              value: useAzureOllama ? ollamaBaseUrl : ''
            }
            {
              name: 'OLLAMA_GENERATE_MODEL'
              value: ollamaGenerateModel
            }
            {
              name: 'OLLAMA_EMBED_MODEL'
              value: ollamaEmbedModel
            }
            {
              name: 'OLLAMA_AUTO_PULL'
              value: useAzureOllama ? 'true' : 'false'
            }
            {
              name: 'OLLAMA_TIMEOUT_SECONDS'
              value: useAzureOllama ? '600' : '240'
            }
            {
              name: 'EVENT_BUS_PROVIDER'
              value: 'azure-service-bus'
            }
            {
              name: 'OPEN_LIBRARY_TIMEOUT_SECONDS'
              value: openLibraryTimeoutSeconds
            }
            {
              name: 'AZURE_SERVICE_BUS_CONNECTION_STRING'
              secretRef: 'service-bus-connection-string'
            }
          ], deployPostgres ? [
            {
              name: 'DATABASE_URL'
              secretRef: 'database-url'
            }
          ] : [], useAzureOpenAI ? [
            {
              name: 'AZURE_OPENAI_ENDPOINT'
              value: azureOpenAIEndpoint
            }
            {
              name: 'AZURE_OPENAI_API_KEY'
              secretRef: 'azure-openai-api-key'
            }
            {
              name: 'AZURE_OPENAI_EMBED_DEPLOYMENT'
              value: azureOpenAIEmbedDeployment
            }
            {
              name: 'AZURE_OPENAI_CHAT_DEPLOYMENT'
              value: azureOpenAIChatDeployment
            }
          ] : [])
          resources: {
            cpu: json(containerCpu)
            memory: containerMemory
          }
        }
      ]
      scale: {
        minReplicas: contains([
          'embedding-worker'
          'recommendation'
        ], service.name) ? 1 : 0
        maxReplicas: 2
      }
    }
  }
}]

resource ollamaApp 'Microsoft.App/containerApps@2024-03-01' = if (deployApps && deployOllama) {
  name: '${normalizedAppName}-ollama'
  location: location
  properties: {
    managedEnvironmentId: acaEnv.id
    configuration: {
      ingress: {
        external: false
        targetPort: 11434
        transport: 'auto'
      }
    }
    template: {
      containers: [
        {
          name: 'ollama'
          image: 'ollama/ollama:latest'
          resources: {
            cpu: json(ollamaCpu)
            memory: ollamaMemory
          }
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 1
      }
    }
  }
}

output acrLoginServer string = acr.properties.loginServer
output logAnalyticsId string = logAnalytics.id
output containerAppsEnvironmentId string = acaEnv.id
output serviceBusNamespace string = serviceBus.name
output postgresServerName string = deployPostgres ? postgresServerName : ''
