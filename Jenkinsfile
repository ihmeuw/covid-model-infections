def conda_dir = "~/.miniconda_snapshot_etl"
def env_name="snapshot-etl-$BUILD_NUMBER"

def cloneRepoToBuild(code_branch, snapshot_branch, etl_branch, share_branch) {
  sh "echo \$(hostname)"
  sh "pwd"
  sh "echo 'mkdir $BUILD_NUMBER'"
  sh "mkdir $BUILD_NUMBER"
  sshagent (credentials: ['covid-jenkins-stash-ssh-access-key-2020-05-06']) {
      sh "echo 'Downloading source code...'"
      sh "git clone --branch $code_branch ssh://git@stash.ihme.washington.edu:7999/scic/covid-snapshot-etl-orchestration.git $BUILD_NUMBER/snapshot-etl/"
      sh "git clone --branch $snapshot_branch ssh://git@stash.ihme.washington.edu:7999/cvd19/covid-input-snapshot.git $BUILD_NUMBER/covid-input-snapshot/"
      sh "git clone --branch $etl_branch ssh://git@stash.ihme.washington.edu:7999/cvd19/covid-input-etl.git $BUILD_NUMBER/covid-input-etl"
      sh "git clone --branch $share_branch https://github.com/ihmeuw/covid-shared.git $BUILD_NUMBER/covid-shared"
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

def poll_jhu(){
   poll_cmd = "$WORKSPACE/$BUILD_NUMBER/snapshot-etl/poll-jhu.sh"
   sh "chmod +x $poll_cmd"
   var = sh(script: "bash $poll_cmd", returnStdout: true).trim()
   sh "echo 'VAR $var'"
   sh "echo If it got this far, the script succeeded and JHU data was present"
   return true
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

      stage ('Cleaning'){
        // Don't run when we want to keep results for debugging
        steps{
          node('qlogin') {
            sh "rm -rf * || true"
            sh "rm -rf $conda_dir || true"

            }
        }
      }


      stage ('Download source code') {
        steps{
          node('qlogin') {
            cloneRepoToBuild("${ORCHESTRATION_BRANCH}", "${SNAPSHOT_BRANCH}", "${ETL_BRANCH}", "${COVID_SHARE_BRANCH}")
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

      stage ('Poll for JHU Data') {
        steps{
          script{
            if(params.WAIT_FOR_JHU){
              got_jhu = false
              catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE'){
                node('qlogin'){
                  got_jhu = poll_jhu()
                  sh "echo 'got_jhu after polling is: $got_jhu'"
                }
              }
              if(got_jhu == false){
                emailext body: 'The Jenkins job did not detect an Automated Update commit in the JHU repository between 10pm PST and now (monitoring the most recent commit). If the data was in fact updated, it will be included in the snapshot-etl that is about to run, otherwise if the JHU data receives an Automated Update after this snapshot-etl has run, it will have to be manually captured. Check console output to view the results:\n\n    $BUILD_URL',
                         to: "${EMAIL_TO}",
                         subject: 'JHU update not found, Proceeding with snapshot-etl'
              }
            }
          }
        }
      }

      stage ('Run snapshot-etl scripts') {
        steps{
              script{
                if(params.PRODUCTION_RUN){
                  node('qlogin') {
                     today = sh(script: 'date +%Y_%m_%d', returnStdout: true).trim()
                     sh "chmod +x $BUILD_NUMBER/snapshot-etl/snapshot-etl.sh"
                     sh "echo 'Snapshot command that will be used: ${SNAPSHOT_CMD} -p $today'"
                     sh "echo 'ETL command that will be used: ${ETL_CMD} -p $today'"
                     ssh_cmd = "$WORKSPACE/$BUILD_NUMBER/snapshot-etl/snapshot-etl.sh $env_name $WORKSPACE/$BUILD_NUMBER $conda_dir \'\"${SNAPSHOT_CMD} -p $today\"\' \'\"${ETL_CMD} -p $today\"\'"
                     sh "echo 'ssh cmd to send is $ssh_cmd'"
                     //TODO: change to qsub later
                     sshagent(['svccovidci-privatekey']) {
                        sh "ssh -o StrictHostKeyChecking=no svccovidci@int-uge-archive-p012.cluster.ihme.washington.edu \"$ssh_cmd\""
                     }

                   }
                }else{
                   node('qlogin') {
                      sh "chmod +x $BUILD_NUMBER/snapshot-etl/snapshot-etl.sh"
                     sh "echo 'Snapshot command that will be used: ${SNAPSHOT_CMD}'"
                      sh "echo 'ETL command that will be used: ${ETL_CMD}'"
                      ssh_cmd = "$WORKSPACE/$BUILD_NUMBER/snapshot-etl/snapshot-etl.sh $env_name $WORKSPACE/$BUILD_NUMBER $conda_dir \'\"${SNAPSHOT_CMD}\"\' \'\"${ETL_CMD}\"\'"
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