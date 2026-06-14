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

var normalizedAppName = toLower(appName)
var nameSuffix = uniqueString(subscription().id, resourceGroup().id, normalizedAppName)
var acrName = take('bookai${replace('${normalizedAppName}${nameSuffix}acr', '-', '')}', 50)
var serviceBusName = take('${normalizedAppName}-${nameSuffix}-bus', 50)
var postgresServerName = take('${normalizedAppName}-${nameSuffix}-pg', 63)
var useAzureOpenAI = !empty(azureOpenAIApiKey) && !empty(azureOpenAIEndpoint) && !empty(azureOpenAIEmbedDeployment) && !empty(azureOpenAIChatDeployment)
var databaseUrl = deployPostgres ? 'postgresql://${postgresAdminUser}:${postgresAdminPassword}@${postgresServerName}.postgres.database.azure.com:5432/${postgresDatabaseName}?sslmode=require' : ''
var serviceBusConnectionString = listKeys(resourceId('Microsoft.ServiceBus/namespaces/authorizationRules', serviceBus.name, 'RootManageSharedAccessKey'), '2024-01-01').primaryConnectionString

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
              value: 'http://${normalizedAppName}-llm-service'
            }
            {
              name: 'BOOK_CATALOG_URL'
              value: 'http://${normalizedAppName}-book-catalog'
            }
            {
              name: 'USER_PROFILE_URL'
              value: 'http://${normalizedAppName}-user-profile'
            }
            {
              name: 'RECOMMENDATION_URL'
              value: 'http://${normalizedAppName}-recommendation'
            }
            {
              name: 'LLM_PROVIDER'
              value: service.name == 'llm-service' && useAzureOpenAI ? 'azure-openai' : 'deterministic'
            }
            {
              name: 'EVENT_BUS_PROVIDER'
              value: 'azure-service-bus'
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
        minReplicas: 0
        maxReplicas: 2
      }
    }
  }
}]

output acrLoginServer string = acr.properties.loginServer
output logAnalyticsId string = logAnalytics.id
output containerAppsEnvironmentId string = acaEnv.id
output serviceBusNamespace string = serviceBus.name
output postgresServerName string = deployPostgres ? postgresServerName : ''
