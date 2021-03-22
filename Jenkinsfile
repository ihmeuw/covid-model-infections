def conda_dir = "~/.miniconda_covid_model_infections"
def env_name="infections-$BUILD_NUMBER"

def cloneRepoToBuild(code_branch) {
  sh "echo \$(hostname)"
  sh "pwd"
  sh "echo 'mkdir $BUILD_NUMBER'"
  sh "mkdir $BUILD_NUMBER"
  // sshagent (credentials: ['svccovidvi-privatekey']) {
  sshagent (credentials: ['jenkins-general']) {
      sh "echo 'Downloading source code...'"
      // sh "git clone --branch $code_branch git@github.com:ihmeuw/covid-model-infections.git $BUILD_NUMBER/covid-model-infections/"
      sh "git clone --branch $share_branch https://github.com/ihmeuw/covid-model-infections.git $BUILD_NUMBER/covid-model-infections/"
      sh "ls -lr $BUILD_NUMBER"
      sh "echo 'Source code downloaded'"
  }
}

def install_miniconda(dir) {
  // It seems that on COVID Jenkins every project installs its own mini conda. Let's follow.
  if (fileExists(dir)) {
      sh "echo miniconda already installed at $dir"
  }else {
      sh "wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
      sh "bash Miniconda3-latest-Linux-x86_64.sh -b -p $dir"
  }
}

pipeline {
  //The Jenkinsfile of ssh://git@stash.ihme.washington.edu:7999/scic/covid-snapshot-etl-orchestration.git

  agent { label 'qlogin' }

  stages{
      stage ('Notify job start'){
         steps{
           emailext body: 'Another email will be send when the job finishes.\n\nMeanwhile, you can view the progress here:\n\n    $BUILD_URL',
                          to: "${EMAIL_TO}",
                          subject: 'Build started in Jenkins: $PROJECT_NAME - #$BUILD_NUMBER'
         }
      }

      stage ('Download source code') {
        steps{
          node('qlogin') {
            cloneRepoToBuild("${JEFFREY_BRANCH}")
            }
           }
      }

      stage ('Install miniconda') {
        steps{
          node('qlogin'){
            install_miniconda(conda_dir)
              }
            }
      }

      
      // stage ('Run snapshot-etl scripts') {
      //   steps{
      //         script{
      //           if(params.PRODUCTION_RUN){
      //             node('qlogin') {
      //                today = sh(script: 'date +%Y_%m_%d', returnStdout: true).trim()
      //                sh "chmod +x $BUILD_NUMBER/snapshot-etl/snapshot-etl.sh"
      //                sh "echo 'Snapshot command that will be used: ${SNAPSHOT_CMD} -p $today'"
      //                sh "echo 'ETL command that will be used: ${ETL_CMD} -p $today'"
      //                ssh_cmd = "$WORKSPACE/$BUILD_NUMBER/snapshot-etl/snapshot-etl.sh $env_name $WORKSPACE/$BUILD_NUMBER $conda_dir \'\"${SNAPSHOT_CMD} -p $today\"\' \'\"${ETL_CMD} -p $today\"\'"
      //                sh "echo 'ssh cmd to send is $ssh_cmd'"
      //                //TODO: change to qsub later
      //                sshagent(['svccovidci-privatekey']) {
      //                   sh "ssh -o StrictHostKeyChecking=no svccovidci@int-uge-archive-p012.cluster.ihme.washington.edu \"$ssh_cmd\""
      //                }

      //              }
      //           }else{
      //              node('qlogin') {
      //                 sh "chmod +x $BUILD_NUMBER/snapshot-etl/snapshot-etl.sh"
      //                sh "echo 'Snapshot command that will be used: ${SNAPSHOT_CMD}'"
      //                 sh "echo 'ETL command that will be used: ${ETL_CMD}'"
      //                 ssh_cmd = "$WORKSPACE/$BUILD_NUMBER/snapshot-etl/snapshot-etl.sh $env_name $WORKSPACE/$BUILD_NUMBER $conda_dir \'\"${SNAPSHOT_CMD}\"\' \'\"${ETL_CMD}\"\'"
      //                 sh "echo 'ssh cmd to send is $ssh_cmd'"
      //                 //TODO: change to qsub later
      //                 sshagent(['svccovidci-privatekey']) {
      //                    sh "ssh -o StrictHostKeyChecking=no svccovidci@int-uge-archive-p012.cluster.ihme.washington.edu \"$ssh_cmd\""
      //                 }
      //              }
      //           }
      //         }
      //     }
      //   }
      }


  post {
       // Currently only email notification is available on COVID Jenkins. If we want to do slack, we will have to
       // coordinate with INFRA to set it up first. It may request server reboot.
       success {
                emailext body: 'Check console output to view the results:\n\n    $BUILD_URL',
                          to: "${EMAIL_TO}",
                          subject: 'Build succeeded in Jenkins: $PROJECT_NAME - #$BUILD_NUMBER'
        }
       failure {
                emailext body: 'Check console output to view the results:\n\n    $BUILD_URL',
                         to: "${EMAIL_TO}",
                         subject: 'Build failed in Jenkins: $PROJECT_NAME - #$BUILD_NUMBER'

            }
    }
}
