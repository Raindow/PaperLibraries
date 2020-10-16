just for test

本地建库后推送

```bash
# 本地库初始
git init
# 本地库添加新文件
git add 文件 或是
git add 文件夹
# 修改描述
git commit -m "想说啥都行"
# 最好尝试拉一次，如果还担心远端把本地覆盖，可以用下面的拉命令
# 这条命令会在本地合并远端的内容
git pull --rebase origin master
# 然后上传，这里我直接传到主分支了
git push -u origin master
```

一般的拉取：

```bash
# git pull origin master
```

一般的提交：

```bash
# 修改、提交前，都要先拉取。。
# git pull --rebase origin master
# 如果有新文件
git add "文件名"
# 修改说明
git commit -m "说明"
# 提交
git push origin master
# 可能需要你提交仓库的账号密码
# Raindow
# hezijishuohua8
```

错误情况统计：

- Changes not staged for commit

	提交时加上参数：-a ，表示新增。

	```bash
	git commit -am "提交说明"
	```
	
- Untracked files

	把文件添加进去

	```bash
	git add ""文件名
	```

	