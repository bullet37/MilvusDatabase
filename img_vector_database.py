from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
from datetime import datetime, timedelta
import logging

class ImgVectorDatabase(): # 管理数据库conncetion 和 Vectorizer
    def __init__(self, connection_name="default"):
        self.connection_name = connection_name
        connections.connect(db_name=self.connection_name, host="localhost", port="19530")
        self.collection_list = utility.list_collections()

    def check_collection(self, collection_name="default"):
        if utility.has_collection(collection_name,timeout=10) and collection_name in self.collection_list:
            #print(f"Collection {collection_name} already exists!")
            return True
        else:
            logging.error("Collection {} does not exist".format(collection_name))
            return False

    def query_entities(self, collection_name="default", query_vector=[], nprobe=10, search_top_k=8,\
                       output_fields=["pk"], chunk_size=10):
        if not self.check_collection(collection_name):
            return None

        collection = self.get_collection(collection_name)
        print(
            f"Number of entities in collection {collection_name}: {collection.num_entities}")  # check the num_entities
        start_time = datetime.now()
        vectors_to_search = list(query_vector)
        search_amount = len(vectors_to_search)

        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": nprobe},  # 从倒排列表中查找 nprobe 个最相近的候选项进行精确的距离计算
        }

        result = collection.search(vectors_to_search, "embeddings", search_params, limit=search_top_k,
                                   output_fields=output_fields)
        end_time = datetime.now()
        logging.info(f"Searching entities used time: {(end_time - start_time).total_seconds()}")
        return result

    def add_entities(self,collection_name, img_vector, data_name,chunk_size=10000):
        if not self.check_collection(collection_name):
            return

        collection = self.get_collection(collection_name)
        start_time = datetime.now()
        total_len = img_vector.shape[0]
        insert_result = None

        if total_len < chunk_size:
            entities = [
                [str(item) for item in data_name],
                img_vector,  # field embeddings, supports numpy.ndarray and list
            ]
            insert_result = collection.insert(entities)
        else:
            pre = start_time
            for i in range(0, total_len, chunk_size):
                entities = [
                    [str(item) for item in data_name[i:i + chunk_size]],
                    img_vector[i:i + chunk_size],  # field embeddings, supports numpy.ndarray and list
                ]
                insert_result = collection.insert(entities)
                now = datetime.now()
                logging.info(f"Inserting {i}th images..... Used time: {(now - pre).total_seconds()}")
                pre = now

        end_time = datetime.now()
        logging.info(insert_result)
        logging.info(f"Inserting entities used time: {(end_time - start_time).total_seconds()}")
        collection.flush()

        index = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128},  # 倒排文件中创建 128 个倒排列表（inverted lists）。较大的 nlist 值有助于提高搜索速度，但会增加索引的内存消耗。
        }
        # expr="random > -14"，表示查询 hello_milvus 中 "random" 属性大于 -14 的实体
        collection.create_index(field_name="embeddings", index_params=index)
        collection.load()



    def update_data(self):
        # Invalid in Milvus for now
        pass

    def delete_data(self):
        pass

    def test_rename_collection(self):
        pass
        # utility.rename_collection(old_collection, new_collection)
        # new_db = "new_db"
        # if new_db not in db.list_database():
        #     print("\ncreate database: new_db")
        #     db.create_database(db_name=new_db)
        # utility.rename_collection(new_collection, new_collection, new_db)
        # print("rename db name end, db:default, collections:", utility.list_collections())
        # db.using_database(db_name=new_db)
        # assert utility.has_collection(new_collection)
        # print("rename db name end, db:", new_db, "collections:", utility.list_collections())
    def get_collection(self, collection_name="default"):
        if self.check_collection(collection_name):
          collection = Collection(collection_name)
          return collection
        else:
          logging.error("Get collection Fail, collection {} does not exist".format(collection_name))
          return None


    def add_collection(self, dim, collection_name="default", primary_key="pk", des="ImgVectorDatabase Schema", \
             consistency_level="Strong"):
        if self.check_collection(collection_name):
            return
        fields = [
            FieldSchema(name=primary_key, dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=500),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim)
            # FieldSchema(name="random", dtype=DataType.DOUBLE),
        ]
        schema = CollectionSchema(fields, des)
        collection = Collection(collection_name, schema, consistency_level=consistency_level)
        assert collection.is_empty is True
        assert collection.num_entities == 0
        assert len(collection.indexes) != 0
        self.collection_list.append(collection_name)
        print(f"Number of entities in collection {collection_name}: {collection.num_entities}")  # check the num_entities

    def drop_collection(self,collection_name):
        if self.check_collection(collection_name):
            self.collection_list.remove(collection_name)
            # collection.drop_index()
            # collection.release()
            utility.drop_collection(collection_name)
            logging.info("Collection {} has been deleted".format(collection_name))
        else:
            logging.error("Drop collection Fail, collection {} does not exist".format(collection_name))
