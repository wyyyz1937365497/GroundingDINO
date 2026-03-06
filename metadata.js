import axios from 'axios'

// Short UUID生成工具
export function generateShortUuid() {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
  let result = ''
  for (let i = 0; i < 8; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length))
  }
  return result
}

// 生成问题编码（带前缀）
export function generateIssueCode() {
  const date = new Date()
  const year = date.getFullYear().toString().slice(-2)
  const month = (date.getMonth() + 1).toString().padStart(2, '0')
  const day = date.getDate().toString().padStart(2, '0')
  const shortId = generateShortUuid()
  return `QI${year}${month}${day}${shortId}`
}

// iS3数字底座元数据结构定义
export const metadataStructures = {
  // 角色表元数据
  roles: {
    "metaTableCode": "quality_roles",
    "tableName": "角色管理表",
    "metaColumnList": [
      {
        "metaColumnCode": "code",
        "columnName": "角色代码",
        "nativeType": "NVARCHAR"
      },
      {
        "metaColumnCode": "name",
        "columnName": "角色名称",
        "nativeType": "NVARCHAR"
      },
      {
        "metaColumnCode": "description",
        "columnName": "角色描述",
        "nativeType": "NVARCHAR"
      },
      {
        "metaColumnCode": "permissions",
        "columnName": "权限列表",
        "nativeType": "TEXT"
      }
    ]
  },

  // 用户表元数据
  users: {
    "metaTableCode": "quality_users",
    "tableName": "用户管理表",
    "metaColumnList": [
      {
        "metaColumnCode": "name",
        "columnName": "用户姓名",
        "nativeType": "NVARCHAR"
      },
      {
        "metaColumnCode": "role",
        "columnName": "用户角色",
        "nativeType": "NVARCHAR"
      },
      {
        "metaColumnCode": "phone",
        "columnName": "联系电话",
        "nativeType": "NVARCHAR"
      },
      {
        "metaColumnCode": "email",
        "columnName": "邮箱地址",
        "nativeType": "NVARCHAR"
      },
      {
        "metaColumnCode": "department",
        "columnName": "所属部门",
        "nativeType": "NVARCHAR"
      },
      {
        "metaColumnCode": "company",
        "columnName": "所属公司",
        "nativeType": "NVARCHAR"
      }
    ]
  },

  // 问题任务表元数据
  issues: {
    "metaTableCode": "quality_issues",
    "tableName": "质量问题任务表",
    "metaColumnList": [
      {
        "metaColumnCode": "code",
        "columnName": "问题编码",
        "nativeType": "NVARCHAR"
      },
      {
        "metaColumnCode": "title",
        "columnName": "问题标题",
        "nativeType": "NVARCHAR"
      },
      {
        "metaColumnCode": "description",
        "columnName": "问题描述",
        "nativeType": "TEXT"
      },
      {
        "metaColumnCode": "status",
        "columnName": "处理状态",
        "nativeType": "NVARCHAR"
      },
      {
        "metaColumnCode": "priority",
        "columnName": "优先级",
        "nativeType": "NVARCHAR"
      },
      {
        "metaColumnCode": "creator",
        "columnName": "发起人ID",
        "nativeType": "NVARCHAR"
      },
      {
        "metaColumnCode": "create_time",
        "columnName": "发起时间",
        "nativeType": "NVARCHAR"
      },
      {
        "metaColumnCode": "mentioned_users",
        "columnName": "@的用户ID列表(逗号分隔)",
        "nativeType": "TEXT"
      },
      {
        "metaColumnCode": "location",
        "columnName": "问题位置",
        "nativeType": "NVARCHAR"
      },
      {
        "metaColumnCode": "category",
        "columnName": "问题分类",
        "nativeType": "NVARCHAR"
      },
      {
        "metaColumnCode": "finishTime",  // deadline 改为 finishTime
        "columnName": "完成时间",   // 截止时间 改成完成时间
        "nativeType": "NVARCHAR"  // 时间类型 改成 NVARCHAR
      },
      {
        "metaColumnCode": "attachments",
        "columnName": "附件文件IDs(逗号分隔)",
        "nativeType": "TEXT",
        "tags": "sys_file"
      }
    ]
  },

  // 问题处理记录表元数据
  comments: {
    "metaTableCode": "quality_comments",
    "tableName": "问题处理记录表",
    "metaColumnList": [
      {
        "metaColumnCode": "issue_code",
        "columnName": "关联问题编码",
        "nativeType": "NVARCHAR"
      },
      {
        "metaColumnCode": "user_id",
        "columnName": "回复用户ID",
        "nativeType": "NVARCHAR"
      },
      {
        "metaColumnCode": "content",
        "columnName": "回复内容",
        "nativeType": "TEXT"
      },
      // {
      //   "metaColumnCode": "action_type",
      //   "columnName": "操作类型",
      //   "nativeType": "NVARCHAR"
      // },
      {
        "metaColumnCode": "mentioned_users",
        "columnName": "@的用户ID列表(逗号分隔)",
        "nativeType": "TEXT"
      },
      // {
      //   "metaColumnCode": "progress",
      //   "columnName": "处理进度",
      //   "nativeType": "NVARCHAR"
      // },
      // {
      //   "metaColumnCode": "next_step",
      //   "columnName": "下一步计划",
      //   "nativeType": "TEXT"
      // },
      {
        "metaColumnCode": "attachments",
        "columnName": "附件文件IDs(逗号分隔)",
        "nativeType": "TEXT",
        "tags": "sys_file"
      },
      {
        "metaColumnCode": "time",   //新增字段，用于描述回复时间
        "columnName": "回复时间",
        "nativeType": "NVARCHAR"
      }
    ]
  },

  // 文件管理表元数据
  files: {
    "metaTableCode": "quality_files",
    "tableName": "文件管理表",
    "metaColumnList": [
      {
        "metaColumnCode": "original_name",
        "columnName": "原始文件名",
        "nativeType": "NVARCHAR"
      },
      {
        "metaColumnCode": "file_type",
        "columnName": "文件类型",
        "nativeType": "NVARCHAR"
      },
      {
        "metaColumnCode": "file_size",
        "columnName": "文件大小",
        "nativeType": "BIGINT"
      },
      {
        "metaColumnCode": "upload_user",
        "columnName": "上传用户ID",
        "nativeType": "NVARCHAR"
      },
      {
        "metaColumnCode": "related_id",
        "columnName": "关联记录ID",
        "nativeType": "NVARCHAR"
      },
      {
        "metaColumnCode": "related_type",
        "columnName": "关联记录类型",
        "nativeType": "NVARCHAR"
      },
      {
        "metaColumnCode": "url",
        "columnName": "文件访问路径",
        "nativeType": "NVARCHAR",
        "tags": "sys_file"
      }
    ]
  },

  // 项目配置表元数据
  project_config: {
    "metaTableCode": "quality_project_config",
    "tableName": "项目配置表",
    "metaColumnList": [
      {
        "metaColumnCode": "project_name",
        "columnName": "项目名称",
        "nativeType": "NVARCHAR"
      },
      {
        "metaColumnCode": "project_code",
        "columnName": "项目编码",
        "nativeType": "NVARCHAR"
      },
      {
        "metaColumnCode": "project_manager",
        "columnName": "项目经理",
        "nativeType": "NVARCHAR"
      },
      {
        "metaColumnCode": "start_date",
        "columnName": "开始时间",
        "nativeType": "DATETIME"
      },
      {
        "metaColumnCode": "end_date",
        "columnName": "结束时间",
        "nativeType": "DATETIME"
      },
      {
        "metaColumnCode": "workflow_config",
        "columnName": "流程配置",
        "nativeType": "TEXT"
      },
      {
        "metaColumnCode": "notification_config",
        "columnName": "通知配置",
        "nativeType": "TEXT"
      }
    ]
  },

  // URL管理表元数据
  url: {
    "metaTableCode": "quality_url",
    "tableName": "URL管理表",
    "metaColumnList": [
      {
        "metaColumnCode": "code",
        "columnName": "问题编码",
        "nativeType": "NVARCHAR"
      },
      {
        "metaColumnCode": "url",
        "columnName": "访问链接",
        "nativeType": "NVARCHAR"
      }
    ]
  }
};

// API工具类
export class MetadataAPI {
  constructor(baseUrl, prjId, headers, folderId = null, fileBaseUrl = null) {
    this.baseUrl = baseUrl;
    this.prjId = prjId;
    this.headers = headers;
    this.folderId = folderId;
    this.fileBaseUrl = fileBaseUrl || "http://43.137.38.138:19000";
  }

  // 创建数据
  async createData(metaTableCode, data) {
    try {
      const response = await axios.post(
        `${this.baseUrl}/data-main/operation/addData`,
        {
          prjId: this.prjId,
          metaTableCode,
          data: Array.isArray(data) ? data : [data]
        },
        { headers: this.headers }
      );
      return response.data;
    } catch (error) {
      console.error('创建数据失败:', error);
      throw error;
    }
  }

  // 创建元数据表
  async createMetaTable(metaTableStructure) {
    try {
      const response = await axios.post(
        `${this.baseUrl}/meta/createMetaTable`,
        {
          prjId: this.prjId,
          ...metaTableStructure
        },
        { headers: this.headers }
      );
      return response.data;
    } catch (error) {
      console.error('创建元数据表失败:', error);
      throw error;
    }
  }

  // 插入数据
  async insertData(metaTableCode, data) {
    try {
      const response = await axios.post(
        `${this.baseUrl}/data-main/operation/addData`,
        {
          prjId: this.prjId,
          metaTableCode,
          data:data
        },
        { headers: this.headers }
      );
      return response.data;
    } catch (error) {
      console.error('插入数据失败:', error);
      throw error;
    }
  }

  // 根据sysId更新一条数据
  async updateDataBySysId(metaTableCode, oneData, sysId) {
      try {
        const response = await axios.post(
          `${this.baseUrl}/data-main/operation/updateDataByCondition`,
          {
            prjId: this.prjId,
            metaTableCode,
            data:[oneData],
            whereConditions:[{"column":"sys_id","logic":"EQUALS","value":sysId}]
          },
          { headers: this.headers }
        );
        return response.data;
      } catch (error) {
        console.error('更新数据失败:', error);
        throw error;
      }
    }

  // 查询数据
  async queryData(metaTableCode, whereConditions = [], pageNumber = 0, pageSize = 0) {
    try {
      const response = await axios.post(
        `${this.baseUrl}/data-main/operation/getDataByCondition`,
        {
          pageNumber,
          pageSize,
          prjId: this.prjId,
          desc: true,
          metaTableCode,
          keyValuePairs: {},
          whereConditions
        },
        { headers: this.headers }
      );
      
      const originalData = response.data.data;
      
      // 如果是质量问题任务表或问题处理记录表，预处理文件信息
      if (metaTableCode === 'quality_issues' || metaTableCode === 'quality_comments') {
        return await this.preprocessFileData(originalData, metaTableCode);
      }
      
      return originalData;
    } catch (error) {
      console.error('查询数据失败:', error);
      throw error;
    }
  }

  // 预处理数据中的文件信息
  async preprocessFileData(responseData, metaTableCode) {
    try {
      console.log('🔄 开始预处理文件数据:', metaTableCode);
      
      // 获取文件映射
      const filesMap = await this.getAllFilesMap();
      
      // 处理数据中的records
      const records = responseData.records || [];
      
      const processedRecords = records.map(record => {
        const processedRecord = { ...record };
        
        // 处理attachments字段
        if (record.attachments) {
          const attachmentIds = this.parseAttachmentIds(record.attachments);
          const attachmentInfos = attachmentIds
            .map(id => filesMap[id])
            .filter(fileInfo => fileInfo); // 只保留找到的文件
          
          // 将attachments替换为文件信息数组
          processedRecord.attachments = attachmentInfos;
          processedRecord._originalAttachments = record.attachments; // 保留原始数据
        }
        
        return processedRecord;
      });
      
      console.log('✅ 文件数据预处理完成，处理了', processedRecords.length, '条记录');
      
      // 返回处理后的数据结构
      return {
        ...responseData,
        records: processedRecords
      };
      
    } catch (error) {
      console.error('❌ 文件数据预处理失败:', error);
      // 预处理失败时返回原始数据
      return responseData;
    }
  }

  // 解析附件ID（统一字符串格式）
  parseAttachmentIds(attachments) {
    return attachments  //返回已经是数组了
    // if (!attachments || typeof attachments !== 'string') return [];
    // return attachments.split(',').map(id => id.trim()).filter(id => id);
  }

  // 更新数据
  async updateData(metaTableCode, data, whereConditions) {
    try {
      const response = await axios.post(
        `${this.baseUrl}/data-main/operation/updateDataByCondition`,
        {
          prjId: this.prjId,
          metaTableCode,
          data,
          whereConditions
        },
        { headers: this.headers }
      );
      return response.data;
    } catch (error) {
      console.error('更新数据失败:', error);
      throw error;
    }
  }

  // 关联文件到数据记录
  async updateFileAddress(metaTableCode, sysId, fileAddress) {
    try {
      const response = await axios.post(
        `${this.baseUrl}/data-main/operation/updateFileAddress`,
        {
          prjId: this.prjId,
          metaTableCode,
          sysId,
          fileAddress
        },
        { headers: this.headers }
      );
      return response.data;
    } catch (error) {
      console.error('关联文件失败:', error);
      throw error;
    }
  }

  // 获取文件列表
  async getFileList(folderId, pageNum = 0, pageSize = 0) {
    try {
      const response = await axios.get(
        `${this.baseUrl}/system/material/list?pageNum=${pageNum}&pageSize=${pageSize}&folderId=${folderId}`,
        { headers: this.headers }
      );
      return response.data;
    } catch (error) {
      console.error('获取文件列表失败:', error);
      throw error;
    }
  }

  // 获取所有文件并建立ID映射
  async getAllFilesMap() {
    try {
      console.log('📁 获取文件夹所有文件列表...');
      const response = await this.getFileList(this.folderId);
      const files = (response.data && response.data.records) || response.records || [];
      
      // 建立文件ID到文件信息的映射
      const filesMap = {};
      
      files.forEach(file => {
        const fileId = file.id;
        if (fileId) {
          // 统一处理文件URL - 在接口返回层面处理
          const processedUrl = this.processFileUrl(file.url || file.downloadUrl || file.fileUrl);
          
          filesMap[fileId] = {
            ...file, // 保留原始数据
            id: fileId,
            original_name: file.original_name || file.name || file.fileName,
            url: processedUrl, // 使用处理后的完整URL
            name: file.name || file.original_name,
            size: file.size || file.fileSize,
            type: file.type || file.fileType,
            uploadTime: file.uploadTime || file.sys_create_time,
          };
          
          // console.log('🔗 文件URL处理:', fileId, file.url || file.downloadUrl || file.fileUrl, '->', processedUrl);
        }
      });
      
      console.log('✅ 文件映射建立完成，共', Object.keys(filesMap).length, '个文件');
      return filesMap;
    } catch (error) {
      console.error('❌ 获取文件列表失败:', error);
      throw error;
    }
  }

  // 统一处理文件URL的方法
  processFileUrl(originalUrl) {
    if (!originalUrl) {
      return null;
    }
    
    // 如果已经是完整URL，直接返回
    if (originalUrl.startsWith('http://') || originalUrl.startsWith('https://')) {
      console.log('🌐 URL已完整:', originalUrl);
      return originalUrl;
    }
    
    // 拼接完整URL
    const fullUrl = this.fileBaseUrl + originalUrl;
    // console.log('🔗 URL拼接:', originalUrl, '+', this.fileBaseUrl, '=', fullUrl);
    return fullUrl;
  }


  // 上传文件
  async uploadFile(file, folderId) {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post(
        `${this.baseUrl}/system/material/uploadFile/${folderId}`,
        formData,
        {
          headers: {
            'X-Access-Key': this.headers['X-Access-Key'],
            'X-Secret-Key': this.headers['X-Secret-Key'],
            'Content-Type': 'multipart/form-data'
          }
        }
      );
      return response.data;
    } catch (error) {
      console.error('上传文件失败:', error);
      throw error;
    }
  }

  // 创建质量问题任务
  async createIssue(issueData, files = []) {
    try {
      console.log('🔄 createIssue 接收到的原始数据:', issueData);
      console.log('📁 需要上传的文件数量:', files.length);
      
      // 先上传文件（如果有）
      let uploadedFileIds = [];
      if (files && files.length > 0) {
        console.log('📤 开始上传文件...');
        for (let file of files) {
          const uploadResult = await this.uploadFile(file, this.folderId);
          uploadedFileIds.push((uploadResult.data && uploadResult.data.id) || uploadResult.id);
          console.log('✅ 文件上传成功:', file.name, '-> ID:', (uploadResult.data && uploadResult.data.id) || uploadResult.id);
        }
      }
      
      // 合并文件ID（已上传的 + 新上传的）
      const allAttachments = [
        ...(Array.isArray(issueData.attachments) ? issueData.attachments : []),
        ...uploadedFileIds
      ];
      
      // 确保包含必要字段
      const taskData = {
        code: issueData.code || generateIssueCode(),
        title: issueData.title,
        description: issueData.description,
        status: issueData.status || 'progress',
        priority: issueData.priority || 'medium',
        creator: issueData.creator,
        creator_name: issueData.creator_name,
        create_time: issueData.create_time || new Date().toISOString(),
        mentioned_users: Array.isArray(issueData.mentioned_users) 
          ? issueData.mentioned_users.join(',') 
          : (issueData.mentioned_users || ''),
        mentioned_users_names: issueData.mentioned_users_names,
        location: issueData.location || '',
        category: issueData.category || '',
        finishTime: issueData.finishTime || '',
        attachments: allAttachments.join(',') // 转为逗号分隔字符串
      };

      console.log('✅ createIssue 转换后的数据:', taskData);
      console.log('📝 mentioned_users 转换:', {
        原始: issueData.mentioned_users,
        类型: Array.isArray(issueData.mentioned_users) ? '数组' : typeof issueData.mentioned_users,
        转换结果: taskData.mentioned_users
      });
      console.log('📎 attachments 转换:', {
        原始: issueData.attachments,
        新上传: uploadedFileIds,
        最终结果: taskData.attachments
      });

      const response = await this.insertData('quality_issues', [taskData]);
      
      // 如果有文件，关联文件到记录
      if (uploadedFileIds.length > 0 && response.data && response.data.sys_id) {
        await this.updateFileAddress('quality_issues', response.data.sys_id, uploadedFileIds.join(','));
        console.log('🔗 文件已关联到记录');
      }
      
      return response;
    } catch (error) {
      console.error('创建问题任务失败:', error);
      throw error;
    }
  }

  // 添加问题处理记录
  async addComment(commentData, files = []) {
    try {
      console.log('🔄 addComment 接收到的原始数据:', commentData);
      console.log('📁 需要上传的文件数量:', files.length);
      
      // 先上传文件（如果有新文件需要上传）
      let uploadedFileIds = [];
      if (files && files.length > 0) {
        console.log('📤 开始上传评论附件...');
        for (let file of files) {
          const uploadResult = await this.uploadFile(file, this.folderId);
          uploadedFileIds.push((uploadResult.data && uploadResult.data.id) || uploadResult.id);
          console.log('✅ 评论附件上传成功:', file.name, '-> ID:', (uploadResult.data && uploadResult.data.id) || uploadResult.id);
        }
      }
      
      // 处理已有的附件ID
      let existingAttachmentIds = [];
      if (commentData.attachments) {
        if (Array.isArray(commentData.attachments)) {
          existingAttachmentIds = commentData.attachments;
        } else if (typeof commentData.attachments === 'string') {
          existingAttachmentIds = commentData.attachments.split(',').filter(id => id.trim());
        }
      }
      
      // 合并文件ID（已有的 + 新上传的）
      const allAttachments = [
        ...existingAttachmentIds,
        ...uploadedFileIds
      ];
      
      const processedData = {
        issue_code: commentData.issue_code,
        user_id: commentData.user_id,
        user_name: commentData.user_name,
        content: commentData.content,
        mentioned_users: Array.isArray(commentData.mentioned_users) 
          ? commentData.mentioned_users.join(',') 
          : (commentData.mentioned_users || ''),
        mentioned_users_names: commentData.mentioned_users_names,
        attachments: allAttachments.join(','), // 转为逗号分隔字符串
        time: commentData.time || new Date().toISOString()
      };

      console.log('✅ addComment 转换后的数据:', processedData);
      console.log('📝 mentioned_users 转换:', {
        原始: commentData.mentioned_users,
        类型: Array.isArray(commentData.mentioned_users) ? '数组' : typeof commentData.mentioned_users,
        转换结果: processedData.mentioned_users
      });
      console.log('📎 attachments 转换:', {
        已有附件: existingAttachmentIds,
        新上传: uploadedFileIds,
        最终结果: processedData.attachments
      });

      const response = await this.insertData('quality_comments', [processedData]);
      
      // 如果有新上传的文件，关联文件到记录
      if (uploadedFileIds.length > 0 && response.data && response.data.sys_id) {
        await this.updateFileAddress('quality_comments', response.data.sys_id, uploadedFileIds.join(','));
        console.log('🔗 评论文件已关联到记录');
      }
      
      return response;
    } catch (error) {
      console.error('添加处理记录失败:', error);
      throw error;
    }
  }
}

 