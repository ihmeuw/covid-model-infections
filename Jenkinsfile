def conda_dir = "~/.miniconda_covid_model_infections"
def env_name="infections-$BUILD_NUMBER"

def cloneRepoToBuild(code_branch) {
  sh "echo \$(hostname)"
  sh "pwd"
  sh "echo 'mkdir $BUILD_NUMBER'"
  sh "mkdir $BUILD_NUMBER"
  sshagent (credentials: ['svccovidvi-privatekey']) {
  // sshagent (credentials: ['jenkins-general']) {
    sh "echo 'Downloading source code...'"
    sh "git clone --branch $code_branch git@github.com:ihmeuw/covid-model-infections.git $BUILD_NUMBER/covid-model-infections/"
    sh "ls -lr $BUILD_NUMBER"
    sh "echo 'Source code downloaded'"
  }
}

pipeline {
  //The Jenkinsfile of ssh://git@stash.ihme.washington.edu:7999/scic/covid-snapshot-etl-orchestration.git

  agent { label 'qlogin' }

  stages {
    stage ('Notify job start') {
      steps {
        emailext  body: 'Another email will be send when the job finishes.\n\nMeanwhile, you can view the progress here:\n\n    $BUILD_URL',
                  to: "${EMAIL_TO}",
                  subject: 'Build started in Jenkins: $PROJECT_NAME - #$BUILD_NUMBER'
      }
    }

    stage ('Download source code') {
      steps {
        node('qlogin') {
          cloneRepoToBuild("${JEFFREY_BRANCH}")
        }
      }
    }

    stage ('Run jenkins scripts') {
      steps{
        script{
          if(params.PRODUCTION_RUN) {
            node('qlogin') {
              today = sh(script: 'date +%Y_%m_%d', returnStdout: true).trim()
              sh "chmod +x $BUILD_NUMBER/covid-model-infections/jeffrey.sh"
              sh "echo 'Command that will be used: ${CMD} -p $today'"
              ssh_cmd = "$WORKSPACE/$BUILD_NUMBER/covid-model-infections/jeffrey.sh $env_name $WORKSPACE/$BUILD_NUMBER $conda_dir \'\"${CMD} -p $today\"\'"
              sh "echo 'ssh cmd to send is $ssh_cmd'"
              //TODO: change to qsub later
              sshagent(['svccovidci-privatekey']) {
                sh "ssh -o StrictHostKeyChecking=no svccovidci@int-uge-archive-p012.cluster.ihme.washington.edu \"$ssh_cmd\""
              }
            }
          } else {
            node('qlogin') {
              today = sh(script: 'date +%Y_%m_%d', returnStdout: true).trim()
              sh "chmod +x $BUILD_NUMBER/covid-model-infections/jeffrey.sh"
              sh "echo 'Command that will be used: ${CMD}'"
              ssh_cmd = "$WORKSPACE/$BUILD_NUMBER/covid-model-infections/jeffrey.sh $env_name $WORKSPACE/$BUILD_NUMBER $conda_dir \'\"${CMD}\"\'"
              sh "echo 'ssh cmd to send is $ssh_cmd'"
              //TODO: change to qsub later
              sshagent(['svccovidci-privatekey']) {
                sh "ssh -o StrictHostKeyChecking=no svccovidci@int-uge-archive-p012.cluster.ihme.washington.edu \"$ssh_cmd\""
              }
            }
          }
        }
      }
    }
  }


  post {
  // Currently only email notification is available on COVID Jenkins. If we want to do slack, we will have to
  // coordinate with INFRA to set it up first. It may request server reboot.
    
    success {
      emailext  body: 'Check console output to view the results:\n\n    $BUILD_URL',
                to: "${EMAIL_TO}",
                subject: 'Build succeeded in Jenkins: $PROJECT_NAME - #$BUILD_NUMBER'
    }
    
    failure {
      emailext  body: 'Check console output to view the results:\n\n    $BUILD_URL',
                to: "${EMAIL_TO}",
                subject: 'Build failed in Jenkins: $PROJECT_NAME - #$BUILD_NUMBER'
    }
  }
}
