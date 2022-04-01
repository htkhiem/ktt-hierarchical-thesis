.. Data Version Control guide

Using DVC with our system
=========================

Our system has builtin support for DVC (Data Version Control), which allows you to (in a personal fork) commit your own datasets and trained models along with the system to your repository for version control, without worrying about the potentially massive file sizes. DVC achieves this by placing placeholder files in the same location of the actual (heavyweight) files, which are text-based and are version control-friendly. You then commit these files and ignore the actual files from your repository, then push the actual files to a remote storage server. As long as the other team members have authorisation to access the same remote, they can then use DVC to 'checkout' those files and have them separately downloaded to their local machine.


