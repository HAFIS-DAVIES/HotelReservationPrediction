pipeline {
    agent any

    environment {
        VENV_DIR = 'venv'
        GCP_PROJECT = "llmops-476223"
        GCLOUD_PATH = "/var/jenkins_home/google-cloud-sdk/bin"
        IMAGE_NAME = "ml-project"
        REGION = "us-central1"
    }

    stages {
        stage('Cloning GitHub Repo') {
            steps {
                script {
                    echo '📦 Cloning repository...'
                    checkout scmGit(
                        branches: [[name: '*/main']],
                        extensions: [],
                        userRemoteConfigs: [[
                            credentialsId: 'github-token',
                            url: 'https://github.com/HAFIS-DAVIES/HotelReservationPrediction.git'
                        ]]
                    )
                }
            }
        }

        stage('Set up Virtual Environment & Dependencies') {
            steps {
                script {
                    echo '⚙️ Setting up Python environment...'
                    sh '''
                        python3 -m venv ${VENV_DIR}
                        . ${VENV_DIR}/bin/activate
                        pip install --upgrade pip
                        pip install -e .
                    '''
                }
            }
        }

        stage('Build & Push Docker Image to GCR') {
            steps {
                withCredentials([file(credentialsId: 'gcp-key', variable: 'GOOGLE_APPLICATION_CREDENTIALS')]) {
                    script {
                        echo '🐳 Building and pushing Docker image...'
                        sh '''
                            export PATH=$PATH:${GCLOUD_PATH}
                            gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}
                            gcloud config set project ${GCP_PROJECT}
                            gcloud auth configure-docker gcr.io --quiet

                            docker build -t gcr.io/${GCP_PROJECT}/${IMAGE_NAME}:latest .
                            docker push gcr.io/${GCP_PROJECT}/${IMAGE_NAME}:latest
                        '''
                    }
                }
            }
        }

        stage('Deploy to Cloud Run') {
            steps {
                withCredentials([file(credentialsId: 'gcp-key', variable: 'GOOGLE_APPLICATION_CREDENTIALS')]) {
                    script {
                        echo '🚀 Deploying to Cloud Run...'
                        sh '''
                            export PATH=$PATH:${GCLOUD_PATH}
                            gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}
                            gcloud config set project ${GCP_PROJECT}

                            gcloud run deploy ${IMAGE_NAME} \
                                --image gcr.io/${GCP_PROJECT}/${IMAGE_NAME}:latest \
                                --platform managed \
                                --region ${REGION} \
                                --allow-unauthenticated \
                                --port 8080 \
                                --memory 512Mi \
                                --timeout 600s
                        '''
                    }
                }
            }
        }
    }

    post {
        success {
            echo '✅ Deployment completed successfully!'
        }
        failure {
            echo '❌ Deployment failed. Check Jenkins logs or Cloud Run logs for details.'
        }
        always {
            echo 'Pipeline execution completed.'
        }
    }
}
