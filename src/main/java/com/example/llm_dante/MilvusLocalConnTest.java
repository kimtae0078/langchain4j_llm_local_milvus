package com.example.llm_dante;

import io.milvus.client.MilvusServiceClient;
import io.milvus.grpc.DataType;
import io.milvus.param.*;
import io.milvus.param.collection.CreateCollectionParam;
import io.milvus.param.collection.FieldType;
import io.milvus.param.collection.LoadCollectionParam;
import io.milvus.param.index.CreateIndexParam;

public class MilvusLocalConnTest {

    public static void main(String[] args) {

        final MilvusServiceClient milvusClient = new MilvusServiceClient(
                ConnectParam.newBuilder()
                        .withHost("localhost")
                        .withPort(19530)
                        //.withToken("root:Milvus")   // id:pw
                        .build()
        );
        System.out.println("milvusClient: "+milvusClient);

        // 필드 생성( PK, segments, vector data를 넣을 수 있는 필드 생성 )
        FieldType pkField = FieldType.newBuilder()
                .withName(CONSTANT.PK_FIELD)
                .withDataType(DataType.VarChar)   // 데이터 타입 명시
                .withMaxLength(36)
                .withPrimaryKey(true)           // PK 지정 여부.
                .withAutoID(true)               // 자동 ID 할당 여부
                .build();
        FieldType txtField = FieldType.newBuilder()
                .withName(CONSTANT.TEXT_FIELD)
                .withDataType(DataType.VarChar) // 데이터 타입 명시
                .withMaxLength(65535)
                .build();
        FieldType vectorField = FieldType.newBuilder()
                .withName(CONSTANT.VECTOR_FIELD)
                .withDataType(DataType.FloatVector) // 데이터 타입 명시
                .withDimension(384)                 // 임베딩 모델의 dimension
                .build();

        // Create Collection
        CreateCollectionParam createCollectionReq = CreateCollectionParam.newBuilder()
                .withCollectionName(CONSTANT.COLLECTION_NAME)   // 필드 이름
                .withDescription("Test Collection Create!")     // 필드 설명
                .withShardsNum(2)                               // 생성할 컬렉션의 샤드 수
                .addFieldType(pkField)
                .addFieldType(txtField)
                .addFieldType(vectorField)
                .withEnableDynamicField(true)
                .build();

        milvusClient.createCollection(createCollectionReq);

        final IndexType INDEX_TYPE = IndexType.FLAT;

        milvusClient.createIndex(
                CreateIndexParam.newBuilder()
                        .withCollectionName(CONSTANT.COLLECTION_NAME)
                        .withFieldName(CONSTANT.VECTOR_FIELD)
                        .withIndexType(INDEX_TYPE)
                        .withMetricType(MetricType.L2)
                        .build()
        );

        milvusClient.loadCollection(
                LoadCollectionParam.newBuilder()
                        .withCollectionName(CONSTANT.COLLECTION_NAME)
                        .build()
        );

    }
}

