python rag.py \
--llm qwen2 \
--qa_file qa400 \
--chunk_size 512 \
--chunk_overlap 100 \
--retriever_top_k 3 \
--reload_vectors_db 1

# python rag.py \
# --llm llama3 \
# --qa_file qa400 \
# --chunk_size 512 \
# --chunk_overlap 100 \
# --retriever_top_k 3 \
# --reload_vectors_db 0

python rag.py \
--llm qwen2 \
--qa_file qa400 \
--chunk_size 512 \
--chunk_overlap 200 \
--retriever_top_k 4 \
--reload_vectors_db 1

# python rag.py \
# --llm llama3 \
# --qa_file qa400 \
# --chunk_size 512 \
# --chunk_overlap 200 \
# --retriever_top_k 4 \
# --reload_vectors_db 0

python rag.py \
--llm qwen2 \
--qa_file qa400 \
--chunk_size 512 \
--chunk_overlap 300 \
--retriever_top_k 4 \
--reload_vectors_db 1


# python rag.py \
# --llm llama3 \
# --qa_file qa400 \
# --chunk_size 512 \
# --chunk_overlap 300 \
# --retriever_top_k 4 \
# --reload_vectors_db 0

python rag.py \
--llm qwen2 \
--qa_file qa400 \
--chunk_size 512 \
--chunk_overlap 300 \
--retriever_top_k 3 \
--reload_vectors_db 1

# python rag.py \
# --llm llama3 \
# --qa_file qa400 \
# --chunk_size 512 \
# --chunk_overlap 300 \
# --retriever_top_k 3 \
# --reload_vectors_db 0

python rag.py \
--llm qwen2 \
--qa_file qa400 \
--chunk_size 1000 \
--chunk_overlap 300 \
--retriever_top_k 4 \
--reload_vectors_db 1


# python rag.py \
# --llm llama3 \
# --qa_file qa400 \
# --chunk_size 1000 \
# --chunk_overlap 300 \
# --retriever_top_k 4 \
# --reload_vectors_db 0

python rag.py \
--llm qwen2 \
--qa_file qa2500 \
--chunk_size 512 \
--chunk_overlap 100 \
--retriever_top_k 3 \
--reload_vectors_db 1

python rag.py \
--llm qwen2 \
--qa_file qa2500 \
--chunk_size 512 \
--chunk_overlap 100 \
--retriever_top_k 5 \
--reload_vectors_db 1

python rag.py \
--llm qwen2 \
--qa_file qa2500 \
--chunk_size 512 \
--chunk_overlap 250 \
--retriever_top_k 5 \
--reload_vectors_db 1

python rag.py \
--llm qwen2 \
--qa_file qa2500 \
--chunk_size 512 \
--chunk_overlap 100 \
--retriever_top_k 10 \
--reload_vectors_db 1


python /home/ubuntu/project/scripts/utils/label_data.py
cd ..
./convert.sh
./eval.sh