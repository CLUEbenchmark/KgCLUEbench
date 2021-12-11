#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/29 22:26
# @Author  : 刘鑫
# @FileName: config.py
# @Software: PyCharm

# 配置

import os

from config import config as pro_config


class Properties(object):
    TASK_DIR = os.path.dirname(__file__)
    PROJECT_DIR = pro_config.PROJECT_DIR
    model_source = os.path.join(PROJECT_DIR, "pretraining_model/chinese_rbtl3_L-3_H-1024_A-16")
    vocab_file = os.path.join(model_source, "vocab.txt")
    bert_config = os.path.join(model_source, "bert_config.json")
    init_checkpoint = os.path.join(model_source, "bert_model.ckpt")


class SimConfig(object):
    max_seq_length = 128
    task_name = "SIM"

    train_data = os.path.join(Properties.TASK_DIR, task_name, "data/train/train.tf_record")
    valid_data = os.path.join(Properties.TASK_DIR, task_name, "data/eval/eval.tf_record")

    train_examples_len = 129981
    valid_examples_len = 13912
    train_batch_size = 40
    valid_batch_size = 40
    num_train_epochs = 20
    eval_start_step = 500
    eval_per_step = 100
    auto_save = 100

    learning_rate = 3e-5
    warmup_proportion = 0.1
    model_out = os.path.join(Properties.TASK_DIR, task_name, "model/")
    training_log = os.path.join(Properties.TASK_DIR, task_name, "log/")
    num_labels = 2  # 采用 True [0,1]  False [1,0]

    class InputFeatures(object):
        """A single set of features of data."""

        def __init__(self,
                     input_ids,
                     input_mask,
                     segment_ids,
                     label_ids,
                     is_real_example=True):
            self.input_ids = input_ids
            self.input_mask = input_mask
            self.segment_ids = segment_ids
            self.label_ids = label_ids
            self.is_real_example = is_real_example

    Attributes = ['主要配音', '具体', '相关学科', '采收时间', '妻子', '所属地区', '医院等级', '平均流量', '地处', '负责', '建筑面积', '始建', '五笔86', '时区',
                  '始建年代', '内设机构',
                  '小说作者', '游戏大小', '配料', '本名', '办公地址', '实施日期', '英文名', '描述', '毒性', '游戏', '现状', '文字', '魔法攻击', '小说类别', '编审',
                  '改编', '费用',
                  '占地', '号码', '名字来源', '连载杂志', '所属行业', '开工时间', '公司性质', '获得荣誉', '专辑歌手', '计算公式', '始建于', '小说性质', '药品名称',
                  '材料', '妹妹',
                  '所属机构', '建筑形式', '单位', '修订时间', '日文', '辅料', '演唱', '专业特点', '交通', '饰演者', '入党时间', '目', '主要', '能力', '管理单位',
                  '纪年', '建校年代',
                  '主治', '出版周期', '建于', '宗教信仰', '经营性质', '公司名称', '内容', '功用', '直径', '印次', '医院', '区划代码', '授予学位', '成立于', '学校',
                  '最高海拔',
                  '区域面积', '下属', '现任职务', '所属区域', '鞋码', '徒弟', '用法用量', '馆藏地点', '功效', '高', '后果', '罗马音', '创办于', '温度', '常住人口',
                  '和声', '校训',
                  '发行公司', '根据地', '启用日期', '医保定点', '开始时间', '其他出版社', '制片人', '丛书', '主要职责', '背景', '持续时间', '拥有者', '命名者',
                  '是否纳入医保', '系统',
                  '赤经', '依托单位', '驻地', '影响', '伤亡情况', '民族族群', '车站代码', '节目类型', '命名人及年代', '美誉', '官位', '诞生时间', '外形', '主角',
                  '节日饮食', '制片地区',
                  '开馆时间', '车站地址', '作者', '曲目数量', '所属领域', '秘书长', '车站等级', '角色', '河长', '采用', '种族', '目标', '隶属单位', '身份',
                  '所属线路', '相关名词',
                  '色彩', '代号', '作品标签', '体裁', '废除时间', '运用', '卡片种类', '主要奖项', '学科', '主要特征', '读音', '学术代表作', '成立年份', '威力',
                  '物理攻击', '作品出处',
                  '耕地', '开发公司', '制片成本', '出身地', '行政代码', '活动时间', '罕见度', '赤纬', '多发群体', '条形码', '长度', '香港首播', '电池容量', '子女',
                  '活跃年代', '地点',
                  '世界排名', '制造商', '种类', '下辖地区', '在校学生', '书号', '登场作品', '包装', '电话区号', '危险性描述', '类型', '分级', '血型', '房间数量',
                  '体重', '软件大小',
                  '上映日期', '产生时间', '视星等', '出道时间', '命名者及年代', '主演', '学生', '运营状态', '参赛运动员', '特征', '性能', '总点击', '网站名称',
                  '物业类别', '批准文号',
                  '所处时代', '邮政区码', '关系', '流经地区', '婚姻状况', '所属朝代', '日期', '景区级别', '创刊时间', '定义', '举例', '尺寸', '举办时间', '总制片人',
                  '竣工时间', '发音',
                  '中文简称', '字数', '途径', '总户数', '在线播放平台', '折射率', '常见病因', '组织', '历史名人', '分布地区', '领导人', '票价', '写作进程', '国家',
                  '代理发行',
                  '起始时间', '艺名', '材质', '集数', '定价', '本质', '社长', '形成时间', '机场代码', '发行商', '制作发行', '制动方式', '旧称', '典故', '年龄',
                  '业务范围',
                  '森林覆盖率', '形成', '风格', '所属单位', '首府', '民系', '现象', '出品', '惯用脚', '软件名称', '完成字数', '前任', '国家代码', '创办地点',
                  '所属国家', '创作年代',
                  '格言', '基本释义', '首都', '父', '传染性', '模式', '校长', '国内刊号', '经济类型', '例如', '宽', '外文名称', '区', '词义', '亚目',
                  '主要农产品', '级别',
                  '流行地区', '执行标准', '武功', '因素', '主办方', '词性', '所在城市', '地理位置', '发布日期', '制作成本', '主题', '文化', '兵器', '印刷时间',
                  '语种', '师承',
                  '监督', '医院类型', '著作', '又叫', '板块', '安全性描述', '系列', '技术', '也称', '应用学科', '朋友', '拼音', '古称', '表达式', '工作内容',
                  '就诊科室', '源自',
                  '主要荣誉', '著名景点', '声音', '软件类型', '嗜好', '近义词', '员工', '文献', '地理标志', '工艺', '游戏目标', '发射时间', '现任校长', '释义',
                  '前身', '首飞时间',
                  '建立', '来源', '属', '兴起时间', '字幕', '继任', '全长', '副作用', '原版名称', '设立', '教练', '符号', '会长', '机型', '影响因素', '主办',
                  '专长', '类别',
                  '三围', '比重', '所属团体', '水溶性', '股票代码', '颁布时间', '代表人物', '字号', '兴趣爱好', '母公司', '现任主席', '开通日期', '最新章节', '编号',
                  '建立时间',
                  '内容主题', '适用领域范围', '学位', '宗教', '中文名', '所在地', '基于', '繁体字', '公式', '法定代表人', '设计师', '身高', '年平均气温', '同义词',
                  '词牌名', '发行',
                  '辖区面积', '主属性', '国土面积', '谥号', '创立人', '现名', '环境', '从事', '片长', '区号', '生日', '总资产', '始于', '亚属', '剧种',
                  '主要景点', '政权',
                  '出道', '主人', '作品名称', '成分', '宗旨', '解释', '原意', '注音', '分布', '创立时间', '进度', '市树', '学生人数', '解散时间', '形象',
                  '耕地总面积', '建造时间',
                  '表示', '标准', '原则', '区域', '创始时间', '行业', '作画', '敌人', '隶属', '大事记', '释义2', '笔画数', '词目', '火车站', '吉祥物', '阶段',
                  '生肖', '发源',
                  '主要成分', '发文机关', '总库容', '人均价格', '所属行政区', '建立者', '逝世地', '演出时间', '占地面积', '小说进度', '源于', '基础', '主场',
                  '粉丝名称', '出生年月',
                  '研制时间', '说明', '物业公司', '译者', '所属类型', '外文队名', '频率', '常见发病部位', '开发者', '相关', '作品体裁', '主要指挥官', '软件授权',
                  '董事长', '形成原因',
                  '组', '涉及领域', '开放时间', '所属年代', '运行环境', '英语', '籍贯', '门票价格', '会员', '日文名', '星座', '幅面', '优势', '制度', '出生地',
                  '居住地', '学名',
                  '主要功能', '存在时间', '修建时间', '亦作', '常见症状', '填词', '学校属性', '电视台', '原因', '建筑风格', '主要分布', '二名法', '通车时间',
                  '发生地点', '球场', '位置',
                  '酒店星级', '丛书系列', '特技', '派别', '玩家人数', '剪辑', '首播频道', '西医学名', '终点', '正文语种', '参考价格', '著名人物', '所属运动队',
                  '发文字号', '功能',
                  '师父', '规模', 'imdb编码', '创办者', '主要原料', '运动项目', '起源于', '军衔', '油耗', '相关作品', '成员', '提出者', '出品方', '出现时间',
                  '依托', '物业费',
                  '特色', '文学体裁', '所获荣誉', '包括', '教授', '版本', '中文别名', '创立者', '每集长度', '引擎类型', '摄制格式', '应用', '笔顺编号', '政治体制',
                  '毕业学校', '笔画',
                  '职称', '发射地点', '饰演', '行政隶属', '发行日期', '旗籍', '通过日期', '纲', '构成', '开播时间', '用处', '编剧', '代表作', '人口密度', '亚组',
                  '新浪微博',
                  '所属洲', '结局', '世纪', '制作公司', '适用人群', '专业', '也叫', '象征', '呼号', '语言', '精灵属性', '介绍', '发明者', '基本解释', '歌曲原唱',
                  '视频格式', '特指',
                  '车身重量', '拍摄日期', '主席', '主要宗教', '是否处方药', '编曲', '班级', '方式', '主要院系', '地理坐标', '主要人物', '外号', '组成部分', '笔顺',
                  '海拔高度', '主持',
                  '去世时间', '晶系', '命名', '分子式', '价值', '注释', '方法', '专辑类别', '面积', '持有者', '条件', '农业人口', '隶属经络', '出版社', '研究方向',
                  '创始人',
                  '片尾曲', '所属类别', '届数', '补充', '制作', '父亲', '注册资金', '唱片销量', '理念', '主要配伍', '实施时间', '服役时间', '菜系', '现任主教练',
                  '恋人', '释文',
                  '产品类型', '高度', '语法', '现居住地', '揭载号', '发生时间', '应用平台', '前提', '颁布单位', '国庆日', '夫君', '根据', '代表色', '颜色',
                  '所属分类', '服务',
                  '民族', '俗称', '座位材质', '角逐赛事', '运营单位', '道路通行', '英文', '定位', '首播平台', '概述', '化学式', '由来', '信仰', '在位时间',
                  '出品年份', '装帧',
                  '建成时间', '投资', '绝招', '代表', '联赛级别', '作用', '软件平台', '所属组织', '职务', '邮编', '轴距', '创建于', '二级学科', '特点', '方向',
                  '选自', '耕地面积',
                  '节日活动', '毕业院校', '通航城市', '形态', '小说类型', '连载期间', '所属系列', '机场', '技术特点', '主要食材', '全程', '有效期', '原作', '释义1',
                  '成立地点', '广义',
                  '现任领导', '首播时间', '保护对象', '简介', '产生', '爱好', '外文名', '闪点', '制作人', '学校地址', '所属专辑', '逝世日期', '病原', '平装',
                  '眼睛颜色', '仓颉',
                  '宠物', '兄弟', '责任编辑', '代表作品', '体长', '职业', '扮演者', '建造者', '游戏类别', '适用领域', '售价', '英文简称', '播放期间', '歌曲时长',
                  '举办单位', '机构性质',
                  '表演者', '发源地', '平台', '设计', '产品', '主要任务', '起源地', '年份', '出自', '出身', '经纬度', '号', '接档', '品牌', '工作单位',
                  '屏幕尺寸', '场上位置',
                  '知名人物', '文号', '起点', '荣誉称号', '主峰', '经营理念', '插画', '始建时间', '拉丁学名', '原作者', '许可证号', '载体', '学科分类', '寓意',
                  '亦称', '作品别名',
                  '结局数', '规划面积', '内涵', '户籍人口', '四角号码', '所属公司', '创办人', '设定时间', '设计单位', '校风', '发起人', '绿化率', '频道', '使用地区',
                  '创建时间',
                  '不良反应', '总投资', '地址', '主要包括', '教学职称', '门票', '主要用药禁忌', '母亲', '用途分类', '兴趣', '施行时间', '详细解释', '技能', '党派',
                  '光泽', '逝世时间',
                  '剧本', '业务', '相关领域', '博士点', '源头', '总建筑面积', '注意', '主要特点', '流域面积', '走向', '瞳色', '口径', '页数', '形状', '发表时间',
                  '阵营', '遗产编号',
                  '行政区类别', '教派', '属性', '沸点', '南北长', '适用', '对白语言', '总部', '运动员慎用', '代码', '是否出版', '类属', '通过会议', '擅长', '工作',
                  '村内资源',
                  '主任', '编导', '难度', '官方语言', '举办地', '郑码', '英文别名', '姓氏', '奖项', '重要事件', '操作系统', '人口', '司掌', '全名', '票房',
                  '获得奖项', '昵称',
                  '工作地点', '星级', '主要支流', '流派', '游戏画面', '主唱', '其他电视台', '时代', '口味', '任期', '隶属机构', '荣誉', '个性', '主要民族',
                  '提出人', '气候',
                  '文献记载', '后期', '成立', '制片', '部首', '所属游戏', '师傅', '卡包', '容纳人数', '属于', '初登场', '喜欢的人', '连载平台', '游戏版本',
                  '开通时间', '节日时间',
                  '归类', '封爵', '反映', '亚种', '原料', '美称', '参战方', '形式', '研发', '标签', '原型', '作品类型', '硕士点', '曾效力球队', '动画制作',
                  '人均收入', '修业年限',
                  '人数', '声优', '必杀技', '党委书记', '包含', '出生', '搭档', '储藏方法', '又称', '时长', '总面积', '出版日期', '主峰海拔', '策划', '创办时间',
                  '英文名称',
                  '地区生产总值', '祖父', '特性', '本科专业', '编辑单位', '丈夫', '原名', '创作者', '游戏语言', '寿命', '长', '专业代码', '目的', '成立日期',
                  '国歌', '概念', '混缩',
                  '校歌', '赛事类型', '起源', '村民小组', '专辑', '例子', '族', '节日起源', '出现', '机构类别', '成就', '冠军', '配音', '中国植物志', '开机时间',
                  '开本', '发现人',
                  '神话体系', '总策划', '拍摄地点', '主管', '洗印格式', '全院人数', '通过时间', '汽车站', '标志', '相关事件', '连载网站', '排名', '备注', '所属品牌',
                  '速度', '曾用名',
                  '利用', '使用者', '剂型', '发布单位', '代表国家队', '所在地区', '儿子', '作曲', '开盘时间', '建设时间', '名人', '总导演', '数量', '父母',
                  '总监制', '姓名', '总监',
                  '年代', '部位', '命名者及时间', '体力', '厚度', '存在', '适宜人群', '其他', '车站类型', '公司类型', '项目', '所属部门', '容积率', '中文名称',
                  '国际域名缩写', '发明人',
                  '感情色彩', '亚纲', '发布时间', '家乡', '建筑类型', '距地距离', '字', '出品公司', '国家/地区', '举办地点', '初次登场', '行政级别', '邮政编码',
                  '物业类型', '规定',
                  '出版', '相关记载', '特长', '原理', '亚族', '音乐', '产业', '卷数', '软件类别', '上映时间', '应用范围', '市长', '主场馆', '楼盘地址', '员工数',
                  '研究内容',
                  '生产商', '歌曲语言', '总科', '相关法律', '演唱者', '指', '轨道倾角', '地位', '成因', '表现形式', '主要产业', '队长', '领域', '上司', '质量',
                  '食性', '简述',
                  '关键词', '车站位置', '主要作用', '主要成就', '农历', '出处', '其他译名', '主要食用功效', '用量', '河口', '所属球队', '上星平台', '区别', '所属国',
                  '配偶',
                  '俱乐部号码', '家族', '血统', '词语解释', '市花', '启用时间', '所在省州', '政府驻地', '操作', '例句', '软件语言', '首发网站', '注册资本', '设立时间',
                  '所属文库',
                  '节日类型', '喜欢的食物', '造价', '学历', '职能', '发行阶段', '效果', '经纪公司', '泛指', '发布机构', '宽度', '产权年限', '环线位置', '参战方兵力',
                  '好友', '所在区域',
                  '运行时间', '网络播放', '危害', '优点', '门', '名称', '作画监督', '来自', '创立', '种', '原创音乐', '死因', '驱动方式', '部门', '适用范围',
                  '实质', '特产',
                  '货币', '创建', '主要作品', '出品时间', '知名校友', '销售', '性质', '用法', '邮发代号', '办学理念', '总笔画数', '终点站', '价格', '又名',
                  '发现者', '症状',
                  '专辑时长', '主要角色', '节日意义', '教师人数', '地区', '城市', '主料', '味道', '主要用途', '省份', '设立机构', '意思', '设计时速', '户数',
                  '国际电话区号', '命名时间',
                  '球场位置', '现居地', '创建人', '球衣号码', '卡片密码', '原画', '混音', '含义', '国家领袖', '院长姓名', '工具', '所在国家', '表现', '朝代',
                  '季数', '配乐',
                  'imdb编号', '年营业额', '贮藏', '应用于', '产权类型', '科', '依据', '副标题', '监制', '学位/学历', '播出时间', '界', '组织性质', '要求',
                  '绰号', '建筑结构',
                  '国家重点学科', '品种', '上市时间', '通过', '启动时间', '吉他', '周期', '获奖', '主要病因', '起因', '连载状态', '科室', '平均海拔', '人均纯收入',
                  '武器', '时间',
                  '正文', '丛书名', '设定地点', '地方', '调料', '总笔画', '主人公', '总人口', '专业方向', '居所', '使用时间', '口头禅', '加速时间', '现任院长',
                  '日语', '领导',
                  '重要性', '运营时间', '游戏类型', '组建时间', '总字数', '续作', '播出状态', '发明时间', '音乐风格', '医生姓名', '原著', '公司', '区位', '人物',
                  '主要内容', '做法',
                  '专辑语言', '纸张', '主办机构', '提出', '归属', '引证解释', '最大城市', '游戏平台', '哥哥', '政党', '重量', '对象', '参加工作', '单行本册数',
                  '主要城市', '陵墓',
                  '公司简称', '最新版本', '所属', '用途', '主要职能', '户型面积', '在校生', '女儿', '退场', '所属组合', '建筑年代', '运营公司', '站址', '角色设计',
                  '结构', '方言',
                  '成立时间', '涵义', '执业地点', '歌手', '站厅形式', '座右铭', '国际刊号', '稀有度', '繁体', '最高时速', '首次登场', '弟弟', '总经理', '运营商',
                  '中文学名', '气候条件',
                  '任务', '局长', '相关文献', '主办单位', '分类', '年降水量', '更名时间', '是否含防腐剂', '离心率', '景点', '遗产级别', '本义', '出生日期', '合同到期',
                  '头衔', '范围',
                  '媒介', '原作名', '车型尺寸', '握拍', '家庭成员', '中文队名', '适用对象', '喜欢的颜色', '城区', '其它译名', '分子量', '偶像', '中医学名', '相对',
                  '记载', '别名',
                  '性格', '开发商', '位于', '出生时间', '文化程度', '行政区划代码', '所属科室', '品质', '网站类型', '核心', '气候类型', '游戏引擎', '追赠', '作词',
                  '机构', '年号',
                  '上线时间', '外观', '地形', '主持人', '摄影', '产地', '门派', '原产国', '政治面貌', '注册号', '歌曲', '姐姐', '批准单位', '大小', '联系电话',
                  '过程', '现居',
                  '国籍', '生产厂商', '英译', '等级', '演出地点', '序号', '学校类型', '格式', '国别', '最高速度', '演员', '韩文名', '范畴', '主要产品', '中医病名',
                  '研究对象',
                  '原产地', '批准时间', '属相', '对应', '发起时间', '传播途径', '主管单位', '车牌代码', '时期', '人物关系', '其他名称', '部外笔画', '亚门', '总部地点',
                  '起源时间',
                  '拍摄时间', '题材', '所处年代', '负责人', '原称', '英文缩写', '研制单位', '主要症状', '音乐类型', '别称', '营业时间', '缩写', '提出时间', '全套枚数',
                  '餐馆名称',
                  '出土地点', '录音', '产生原因', '下辖', '好处', '鼓手', '统一书号', '发色', '首播', '分为', '品质特点', '介质', '使用范围', '家人', '笔名',
                  '熔点', '站台形式',
                  '管理机构', '一级学科', '设置时间', '示例', '状态', '刊行册数', '唱片公司', '发行时间', '所属水系', '官方网站', '统一汉字', '主题曲', '用于',
                  '河流面积', '公布时间',
                  '所属学科', '口号', '播出平台', '系', '书名', '地域', '性状', '发现时间', '作品', '结束时间', '教学班', '出品人', '密度', '主要适用症',
                  '运营方式', '规格',
                  '工作原理', '反义词', '汉语名称', '播出频道', '五笔', '设计者', '主编', '人口数量', '比赛项目', '君主', '组织门派', '简称', '同伴', '提供',
                  '爵位', '服务对象',
                  '原籍', '亚科', '中文全名', '土地面积', '相关书籍', '楼盘名', '水域面积', '临床职称', '馆藏精品', '院系设置', '相对密度', '研究', '总部地址',
                  '精灵序号', '年级',
                  '教职工', '更新时间', '应用领域', '官职', '适用于', '版次', '首播电视台', '半长轴', '标准座位数', '导演', '出版期间', '发病部位', '研究领域',
                  '软件版本', '外貌',
                  '事件', '缺点', '药品类型', '语系', '享年', '组成', '海拔', '实例', '职位', '称号', '谱曲', '授权状态', '创作时间', '主要营养成分', '推出时间',
                  '相关人物', '总裁',
                  '出版时间', '庙号', '评价', '性别', '博士后流动站', '注意事项', '途经线路', '职责', '区长', '分布区域', '危险性符号', '汉语拼音', '遗产类别', '简写',
                  '出土时间',
                  '硬度', '粉丝名', '制作工艺', '封号', '承办单位', '举行时间', '结果', '行政区域', '出道日期', '主要成员', '医院地址', '历史', '创造者', '型号',
                  '公司口号', '车站编号',
                  '分辨率', '发行地区', '党员', '出版地', '处理器', '院长', '景点级别', '全称', '保护级别', '油箱容积', '经营范围', '祖籍', '创建者', '发展',
                  '主管部门', '开业时间',
                  '市委书记', '意义', '医院名称', '重建时间']


class NerConfig(object):
    max_seq_length = 128
    label_list = ["[Padding]", "[##WordPiece]", "[CLS]", "[SEP]", "B-NP", "I-NP", "O"]
    train_data = os.path.join(Properties.TASK_DIR, "NER/data/train/train.tf_record")
    valid_data = os.path.join(Properties.TASK_DIR, "NER/data/eval/eval.tf_record")

    train_examples_len = 18000
    valid_examples_len = 2000
    train_batch_size = 40
    valid_batch_size = 40
    num_train_epochs = 5
    eval_start_step = 500
    eval_per_step = 50
    auto_save = 50

    learning_rate = 3e-5
    warmup_proportion = 0.1
    model_out = os.path.join(Properties.TASK_DIR, "NER/model/")
    training_log = os.path.join(Properties.TASK_DIR, "NER/log/")
    num_labels = len(label_list)

    class InputFeatures(object):
        """A single set of features of data."""

        def __init__(self,
                     input_ids,
                     input_mask,
                     segment_ids,
                     label_ids,
                     is_real_example=True):
            self.input_ids = input_ids
            self.input_mask = input_mask
            self.segment_ids = segment_ids
            self.label_ids = label_ids
            self.is_real_example = is_real_example

class LstmCRFConfig(object):
    lstm_size=128
    cell="lstm"
    num_layers=1
    droupout_rate=0.5

    max_seq_length = 32
    label_list = ["[Padding]", "[##WordPiece]", "[CLS]", "[SEP]", "B-NP", "I-NP", "O"]
    train_data = os.path.join(Properties.TASK_DIR, "NER/data/train/train.tf_record")
    valid_data = os.path.join(Properties.TASK_DIR, "NER/data/eval/eval.tf_record")

    train_examples_len = 18000
    valid_examples_len = 2000
    train_batch_size = 40
    valid_batch_size = 40
    num_train_epochs = 30
    eval_start_step = 500
    eval_per_step = 50
    auto_save = 50

    learning_rate = 3e-5
    warmup_proportion = 0.1
    model_out = os.path.join(Properties.TASK_DIR, "NER_BERT_LSTM_CRF/model/")
    training_log = os.path.join(Properties.TASK_DIR, "NER_BERT_LSTM_CRF/log/")
    num_labels = len(label_list)

    class InputFeatures(object):
        """A single set of features of data."""

        def __init__(self,
                     input_ids,
                     input_mask,
                     segment_ids,
                     label_ids,
                     is_real_example=True):
            self.input_ids = input_ids
            self.input_mask = input_mask
            self.segment_ids = segment_ids
            self.label_ids = label_ids
            self.is_real_example = is_real_example